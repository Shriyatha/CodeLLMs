"""Service for interacting with Large Language Models (LLMs).

This module provides functionality to generate hints, explain errors, and interact
with LLM APIs for educational purposes.
"""
import json
import re
import types
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from app.services.mlflow_logger import MLFlowLogger
from configs.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

# Create __init__.py if it doesn't exist to define the package
init_path = Path(__file__).parent / "__init__.py"
if not init_path.exists():
    init_path.touch()

logger = MLFlowLogger()


class HintLevel(Enum):
    """Levels of hints."""

    GENERAL = 1
    DIRECTIONAL = 2
    SPECIFIC = 3


class LLMService:
    """Service for interacting with Large Language Models."""

    MIN_BULLET_POINT_WORDS = 2
    MAX_HINTS = 3  # Constant for magic number 3
    EMPTY_RESPONSE_MSG = "Empty response content"
    LLM_API_ERROR_MSG = "LLM API error"
    PROCESS_RESPONSE_ERROR = "Could not process LLM response"

    def __init__(self) -> None:
        """Initialize the LLMService with configuration and clients."""
        self.ollama_base_url = Config.OLLAMA_HOST.rstrip("/")
        self.default_model = Config.DEFAULT_LLM
        self.timeout = httpx.Timeout(Config.LLM_TIMEOUT)
        self.client = httpx.AsyncClient()
        self.mlflow_logger = MLFlowLogger
        self.max_retries = 3

    async def __aenter__(self) -> "LLMService":
        """Asynchronous context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Asynchronous context manager exit."""
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _generate(
        self, prompt: str, system_message: str | None = None, temperature: float = 0.3,
    ) -> str:
        """Generate text from the LLM."""
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.default_model,
                    "messages": messages,
                    "options": {"temperature": temperature},
                    "stream": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            try:
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                if not content:
                    raise ValueError(self.EMPTY_RESPONSE_MSG)
                return content
            except json.JSONDecodeError:
                text = response.text.strip()
                if text:
                    return text
                raise ValueError(self.EMPTY_RESPONSE_MSG) from None
            else:
                return content

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.log_llm_interaction(
                prompt={"prompt": prompt, "system_message": system_message},
                response=error_msg,
                metadata={"error": True, "status_code": e.response.status_code},
            )
            api_error = f"{self.LLM_API_ERROR_MSG}: {e.response.text[:200]}"
            raise RuntimeError(api_error) from e
        except (ValueError, RuntimeError) as e:
            error_msg = f"Generation failed: {e!s}"
            logger.log_llm_interaction(
                prompt={"prompt": prompt, "system_message": system_message},
                response=error_msg,
                metadata={"error": True, "exception": str(e)},
            )
            raise RuntimeError(self.PROCESS_RESPONSE_ERROR) from e

    async def get_progressive_hints(
        self, problem: str, code: str, current_level: int,
    ) -> tuple[list[str], int]:
        """Generate tiered hints that gradually guide toward solution."""
        prompt = (
            "You are a **friendly and knowledgeable programming tutor** helping a "
            "student who is learning to solve coding problems. Your job is to "
            "provide **exactly 3 helpful hints** that guide the student toward "
            "the correct solution, without giving away the full answer or "
            "writing code.\n\n"
            f"Problem Statement:\n{problem.strip()}\n\n"
            f"Student's Current Code:\n{code.strip()}\n\n"
            "Your task is to generate exactly 3 progressive hints to help the "
            "student improve or complete their solution.\n\n"
            "Follow these guidelines:\n"
            "- DO NOT solve the problem or provide code.\n"
            "- Each hint should gradually increase in specificity.\n"
            "- Do not repeat previous hint information in the next level.\n"
            "- Be concise and encouraging.\n\n"
            "Use this structure:\n"
            "1. [Level 1 - GENERAL]: High-level guidance or concept\n"
            "2. [Level 2 - DIRECTIONAL]: Point the student toward a part of the "
            "problem or technique\n"
            "3. [Level 3 - SPECIFIC]: Describe a specific change or strategy, but NOT "
            "a code solution\n\n"
            "Example:\n"
            "1. Think about how you can use a loop to iterate over the input.\n"
            "2. Consider using a `for` loop to process each element in the list.\n"
            "3. You might need to check if each element is even and keep a count.\n\n"
            "Now generate your 3 hints for the student's code.\n"
            "Format your response EXACTLY like this:\n"
            "1. [First hint text]\n"
            "2. [Second hint text]\n"
            "3. [Third hint text]"
        )

        try:
            response = await self._generate(
                prompt,
                system_message=(
                    "You are a programming tutor. Provide only the 3 numbered hints "
                    "in the specified format."
                ),
            )

            logger.log_llm_interaction(
                prompt={
                    "problem": problem,
                    "code": code,
                    "current_level": current_level,
                },
                response=response,
                metadata={"method": "get_progressive_hints"},
            )

            hints = []
            for line in response.split("\n"):
                match = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if match and len(hints) < self.MAX_HINTS:
                    hints.append(match.group(1))

            if len(hints) < self.MAX_HINTS:
                hints.extend(self._get_fallback_hints()[len(hints) : self.MAX_HINTS])

            return hints[: self.MAX_HINTS], self.MAX_HINTS

        except (ValueError, RuntimeError) as e:
            error_msg = f"Hint generation failed: {e!s}"
            logger.log_llm_interaction(
                prompt={
                    "problem": problem,
                    "code": code,
                    "current_level": current_level,
                },
                response=error_msg,
                metadata={"error": True, "exception": str(e)},
            )
            return self._get_fallback_hints(), self.MAX_HINTS

    async def explain_errors(
        self,
        error: str,
        code: str,
        problem: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """Generate structured error explanation with actionable fixes."""
        prompt = (
            "You are a helpful and knowledgeable programming tutor.\n\n"
            "Your task is to carefully analyze a code snippet and its associated "
            "error message. Then, explain the error to a beginner student and "
            "suggest actionable ways to fix it.\n\n"
            "---\n"
            f"Programming Language: {language}\n\n"
            f"Problem Description:\n{problem}\n\n"
            f"Error Message:\n{error}\n\n"
            f"Problematic Code:\n{code}\n\n"
            "Analyze this error and provide:\n\n"
            "1. **Error Type Classification**:\n"
            '   Classify the error into a specific type (e.g., "IndexError", '
            '"TypeError", "SyntaxError"). Identify the exact issue based on the '
            "error message.\n"
            '   - Example: If the error message is "IndexError: list index out of '
            'range", classify it as "IndexError".\n\n'
            "2. **Clear Explanation of Why it Occurred**:\n"
            "   Explain in plain language why the error happened. Focus on the root "
            "cause and avoid technical jargon.\n"
            '   - Example: For an "IndexError", the explanation might be:\n'
            '     _"The error occurred because you\'re trying to access an index in '
            "the list that doesn't exist. Lists in Python are zero-indexed, meaning "
            "the first item in the list is at index 0. You're trying to access index "
            '5, but the list only has 3 elements, so index 5 is out of range."_ \n\n'
            "3. **Suggested Fixes** (2-3 bullet points):\n"
            "   Provide 2 to 3 actionable suggestions that will help the student fix "
            "the error. Do not provide the exact code but suggest logical steps to "
            "fix it.\n"
            '   - Example:\n'
            '     - "Ensure the index is within the bounds of the list by checking '
            'its length using `len()`. For example, `if index < len(list):`."\n'
            '     - "Use a `try-except` block to handle situations where an invalid '
            'index might be accessed."\n'
            '     - "Double-check the loop ranges and ensure that the index being '
            'accessed is valid for the length of the list."\n\n'
            "4. **Relevant Line Number if Apparent**:\n"
            "   Identify the line number where the error occurred, if possible. If "
            "the error message or stack trace gives a specific line, point it out. "
            "If the line number is not clear, return `null`.\n"
            '   - Example: If the code `my_list[5]` is on line 8 and causes an error, '
            'output `"relevant_line": 8`.\n\n'
            "5. **Common Mistakes That Lead to This Error**:\n"
            "   List common mistakes that often lead to the error in question. These "
            "could include misunderstandings like forgetting that list indices start "
            "at 0, or not checking if a value exists before trying to access it.\n"
            '   - Example for an `IndexError`:\n'
            '     - "Forgetting that list indices start at 0, not 1."\n'
            '     - "Trying to access an index that exceeds the list\'s length."\n'
            '     - "Using hardcoded index values instead of dynamically checking the '
            'valid range."\n\n'
            "⚠ Do NOT include extra commentary, code, or preamble.\n\n"
            "Format your response as JSON with these keys:\n"
            "{\n"
            '    "error_type": "",\n'
            '    "explanation": "",\n'
            '    "suggested_fixes": [],\n'
            '    "relevant_line": null,\n'
            '    "common_mistakes": []\n'
            "}"
        )

        try:
            response = await self._generate(
                prompt,
                system_message=f"Explain {language} errors in simple terms. Return valid JSON only.",
                temperature=0.3,  # Keep responses focused
            )

            # Clean response by removing markdown code blocks if present
            clean_response = response.replace("```json", "").replace("```", "").strip()

            result = json.loads(clean_response)
            return {
                "error_type": result.get("error_type", self._extract_error_type(error)),
                "explanation": result.get("explanation", "No explanation provided"),
                "suggested_fixes": result.get("suggested_fixes", []),
                "relevant_line": result.get("relevant_line", self._find_relevant_line(error)),
                "common_mistakes": result.get("common_mistakes", []),
            }

        except json.JSONDecodeError:
            return self._parse_unstructured_error_response(response, error)
        except (BaseException, Exception) as e:
            return {
                "error_type": self._extract_error_type(error),
                "explanation": f"Error occurred during analysis: {e!s}",
                "suggested_fixes": ["Review the error message carefully"],
                "relevant_line": self._find_relevant_line(error),
                "common_mistakes": [],
            }

    async def analyze_optimizations(
        self, code: str, problem: str, language: str,
    ) -> dict:
        """Analyzes the provided code for optimization opportunities using an LLM.

        Args:
            code: The code to be analyzed (as a string).
            problem: A description of the problem the code(as a string).
            language: The programming language of the code (as a string).

        Returns:
            A dictionary containing the analysis results, including complexity,
            suggestions, and a potential refactored code snippet.

        """
        prompt = f""""You are a highly skilled code reviewer with strong algorithm "
    "understanding. Your task is to analyze the provided solution and "
    "offer detailed suggestions for improvement. Your review should cover "
    "various aspects of the code, including time and space complexity, "
    "optimization opportunities, readability, best practices, edge cases, "
    "and clear explanations. You should provide actionable feedback for a "
    "beginner developer and suggest ways to improve the solution.\n\n"
    "Problem Description:\n"
    f"{problem.strip()}\n\n"
    "Current Solution:\n"
    f"```{language.lower()}\n"
    f"{code.strip()}\n"
    "Your analysis should include the following key components:\n\n"
    "### 1. **Current Time and Space Complexity**:\n"
    "Analyze the time and space complexity of the provided solution. "
    "Focus on understanding how the code's efficiency scales with the "
    "size of the input. Provide an estimation of both time and space "
    "complexity using Big-O notation (e.g., O(n), O(n²), O(log n), etc.). "
    "The goal is to determine the computational and memory cost of the "
    "current implementation.\n\n"
    "- **Time Complexity**: How does the runtime of the code change as the "
    "input size increases? Are there any nested loops or recursive calls "
    "that increase the time complexity? Describe the relationship between "
    "the input size and the runtime.\n"
    "- **Space Complexity**: How much memory does the code use relative to "
    "the size of the input? Are there any data structures or variables that "
    "increase space usage as the input grows?\n\n"
    "Example:\n"
    '"The current solution has a time complexity of O(n²) because of the '
    'nested loops. The space complexity is O(n) due to the additional '
    'array storing intermediate results."\n\n'
    "### 2. **Suggested Optimizations for Time and Space Complexity**:\n"
    "Identify opportunities to optimize both the time and space complexity. "
    "Suggest ways to improve the performance of the code by reducing its "
    "time and/or space complexity. Consider techniques such as:\n\n"
    "- Using more efficient data structures (e.g., hash maps, heaps, trees).\n"
    "- Rewriting parts of the algorithm to reduce the number of iterations.\n"
    "- Optimizing recursive calls or loops to lower time complexity.\n"
    "- Minimizing memory usage by reusing data structures or reducing the "
    "use of auxiliary variables.\n\n"
    "Example:\n"
    '"Consider using a hash set to track previously seen elements, which '
    'can reduce the time complexity from O(n²) to O(n) in this solution."\n\n'
    "### 3. **Readability Improvements**:\n"
    "Provide suggestions for improving the clarity and readability of the "
    "code. These suggestions should help both beginner and experienced "
    "developers understand the code with ease. Some things to consider "
    "include:\n\n"
    "- Renaming variables and functions for better clarity (e.g., changing "
    "`a` to `itemList` or `process_data` to `process_and_filter_data`).\n"
    "- Breaking long functions into smaller, more manageable ones to "
    "promote code modularity.\n"
    "- Adding comments and documentation to explain the purpose of code "
    "blocks or complex logic.\n"
    "- Ensuring consistent formatting and indentation for easier "
    "understanding.\n\n"
    "Example:\n"
    '"Renaming the variable `a` to `itemList` makes the code more '
    'readable and self-explanatory."\n\n'
    "### 4. **Best Practice Recommendations**:\n"
    "Suggest any industry-standard best practices or guidelines that the "
    "developer should follow. These could be related to:\n\n"
    "- Coding style (e.g., PEP8 for Python).\n"
    "- Use of built-in functions and libraries.\n"
    "- Writing modular, maintainable, and reusable code.\n"
    "- Avoiding common coding pitfalls and anti-patterns (e.g., using global "
    "variables, hardcoding values).\n"
    "- Ensuring proper error handling and input validation.\n\n"
    "Example:\n"
    '"Consider using Python\'s list comprehension feature to improve '
    'readability and efficiency when filtering items from a list."\n\n'
    "### 5. **Important Edge Cases**:\n"
    "Identify any edge cases that the current solution may not handle "
    "effectively. Edge cases are inputs or conditions that could break the "
    "code or lead to incorrect results. Consider edge cases such as:\n\n"
    "- Empty inputs (e.g., empty lists, strings).\n"
    "- Large inputs (e.g., huge arrays or files).\n"
    "- Special values (e.g., negative numbers, zeros, or null values).\n"
    "- Boundary cases (e.g., input that's just above or below a threshold).\n\n"
    "For each identified edge case, explain how the code can be modified "
    "to handle it more effectively.\n\n"
    "Example:\n"
    '"Ensure the solution handles empty arrays and negative values as input '
    'to avoid potential errors."\n\n'
    "### 6. **Clear Explanation of the Optimizations**:\n"
    "Provide a clear and concise explanation of why each suggested "
    "optimization will improve the code. For each optimization, describe:\n\n"
    "- How it will reduce time or space complexity.\n"
    "- How it will improve code readability or maintainability.\n"
    "- Why it's a good practice in terms of efficiency or scalability.\n\n"
    "Make sure the reasoning behind each suggestion is easy to understand "
    "for the user.\n\n"
    "Example:\n"
    '"Using a dictionary to store the count of each element reduces the '
    'time complexity to O(n), improving performance significantly for '
    'larger input sizes."\n\n'
    "### 7. **Relevant Code Snippet**:\n"
    "Optionally, provide a refactored version of the code that "
    "incorporates the suggested optimizations. This will help the student "
    "or developer understand how to implement the changes and improve the "
    "solution. You can refactor the code for:\n\n"
    "- Improved time/space complexity.\n"
    "- Improved readability and modularity.\n"
    "- Better handling of edge cases.\n\n"
    "Example:\n"
    '"Here\'s a refactored version of the code with the suggested '
    'optimizations applied."\n\n'
    "---\n"
    "Format your response as valid JSON with these exact keys:\n"
    ""
    '  "current_complexity": {{"time": "", "space": ""}},\n'
    '  "suggested_complexity": {{"time": "", "space": ""}},\n'
    '  "optimization_suggestions": [],\n'
    '  "readability_suggestions": [],\n'
    '  "best_practice_suggestions": [],\n'
    '  "edge_cases": [],\n'
    '  "explanation": "",\n'
    '  "code_snippet": ""\n'
    ""
"""
        try:
            response = await self._generate(
                prompt,
                system_message=f"""You are a senior {language} engineer reviewing code.
                Return valid JSON with all requested fields.""",
            )

            logger.log_llm_interaction(
                prompt={"code": code, "problem": problem, "language": language},
                response=response,
                metadata={"method": "analyze_optimizations"},
            )

            return self._parse_optimization_response(response, code)

        except httpx.NetworkError as e:  # Example: Catch a specific network error
            error_msg = f"Network error during optimization analysis: {e!s}"
            logger.log_llm_interaction(
                prompt={"code": code, "problem": problem, "language": language},
                response=error_msg,
                metadata={"error": True, "exception": str(e)},
            )
            return self._parse_unstructured_optimization_response("", code)
        except json.JSONDecodeError as e: # Example: Catch JSON parsing error
            error_msg = f"Error decoding LLM response: {e!s}"
            logger.log_llm_interaction(
                prompt={"code": code, "problem": problem, "language": language},
                response=error_msg,
                metadata={"error": True, "exception": str(e)},
            )
            return self._parse_unstructured_optimization_response("", code)

    def _parse_complexity(self,
                          complexity_data: dict[str, str | Any]) -> dict[str, str]:
        """Extract and format complexity information."""
        return {
            "time": str(complexity_data.get("time", "")),
            "space": str(complexity_data.get("space", "")),
        }

    def _parse_list_field(self, data: list[str] | str | None) -> list[str]:
        """Parse a field expected to be a list of strings."""
        if isinstance(data, list):
            return [str(item) for item in data if item]
        if data:
            return [str(data)]
        return []

    def _parse_structured_optimization_response(
        self, response: str, code: str,
    ) -> dict[str, Any]:
        """Parse a structured JSON response for optimization suggestions."""
        default_response = {
            "current_complexity": self._estimate_complexity(code),
            "suggested_complexity": {"time": "", "space": ""},
            "optimization_suggestions": [],
            "readability_suggestions": [],
            "best_practice_suggestions": [],
            "edge_cases": [],
            "explanation": "",
            "code_snippet": "",
        }
        result = json.loads(response)
        validated = default_response.copy()

        if "current_complexity" in result:
            validated["current_complexity"] = self._parse_complexity(
                result["current_complexity"],
            )

        if "suggested_complexity" in result:
            validated["suggested_complexity"] = self._parse_complexity(
                result["suggested_complexity"],
            )

        list_fields = [
            "optimization_suggestions",
            "readability_suggestions",
            "best_practice_suggestions",
            "edge_cases",
        ]
        for field in list_fields:
            if field in result:
                validated[field] = self._parse_list_field(result[field])

        if result.get("explanation"):
            validated["explanation"] = str(result["explanation"])

        if result.get("code_snippet"):
            validated["code_snippet"] = str(result["code_snippet"])
        else:
            validated["code_snippet"] = self._extract_code_snippet(response)

        return validated

    def _parse_optimization_response(
        self, response: str, code: str,
    ) -> dict:
        """Parse the LLM's response for code optimization suggestions."""
        try:
            try:
                json_match = re.search(
                    r"```(?:json)?\n(.*?)\n```", response, re.DOTALL,
                )
                if json_match:
                    return self._parse_structured_optimization_response(
                        json_match.group(1), code,
                    )
                return self._parse_structured_optimization_response(response, code)
            except json.JSONDecodeError:
                return self._parse_unstructured_optimization_response(response, code)
        except (NameError, BaseException):
            return self._parse_unstructured_optimization_response(response, code)


    def _parse_unstructured_optimization_response(
        self, text: str, code: str,
    ) -> dict:
        return {
            "current_complexity": self._extract_complexity(text, "Current")
            or self._estimate_complexity(code),
            "suggested_complexity": self._extract_complexity(text, "Suggested")
            or {"time": "", "space": ""},
            "optimization_suggestions": self._extract_section(text, "Optimization"),
            "readability_suggestions": self._extract_section(text, "Readability"),
            "best_practice_suggestions": self._extract_section(
                text, "Best Practice",
            ),
            "edge_cases": self._extract_section(text, "Edge Case"),
            "explanation": self._extract_first_paragraph(text),
            "code_snippet": self._extract_code_snippet(text),
        }

    async def generate_conceptual_steps(self, problem: str) -> list[str]:
        """Break a coding problem into logical, high-level conceptual steps."""
        prompt = f"""You are helping a student solve this programming problem:

    {problem.strip()}

    Your task is to break this problem into **3 to 5 high-level conceptual
    steps** that outline the thought process necessary to solve it **before**
    writing any code. The steps should focus on the **logical planning** and
    **approach** needed to understand and tackle the problem, without any
    reference to syntax, variables, or code implementation.
    ### Instructions:
    1. **Focus on abstraction**: Think about the **mental framework** required
       to solve the problem, such as identifying the problem's goals, required
       operations, and expected outputs.
    2. **Logical flow**: Each step should build on the previous one. The steps
       must follow a natural progression and lead the student toward a
       structured approach to solve the problem.
    3. **Clarity and simplicity**: Each step should be **one sentence long** and
       **easy to understand**. Keep it simple, with no jargon or
       coding-specific terms.
    4. **Avoid syntax and code details**: Do **not** include any references to
       specific programming languages, code syntax, variable names, or
       functions.
    5. **Planning and problem-solving**: Focus solely on how to approach the
       problem logically. This is about **planning** the solution before actual
       coding.
    6. **Edge cases**: Think about any potential edge cases or special
       conditions that could affect the solution, but do not attempt to solve
       them yet. Just consider them as part of the planning phase.

    Each step must follow these detailed guidelines:

    1. **Be exactly one sentence long**
       - Describe a single, focused thought or action in one clear sentence.
       - Avoid combining multiple ideas — keep each step concise and easy to
         understand.

    2. **Build logically on the previous step**
       - The steps must follow a logical sequence, starting from
         understanding the problem and progressing toward designing a solution.
       - Each step should naturally lead into the next to guide the student's
         thought process.

    3. **Avoid any code, syntax, or variable names**
       - Do not use programming constructs like loops, arrays, dictionaries,
         function names, or specific syntax.
       - The focus must remain purely on conceptual reasoning, not
         implementation details.

    4. **Focus only on thought process and planning**
       - Help the student think about what they need to figure out, organize,
         or decide before they start coding.
       - Emphasize analysis, reasoning, or decision-making, such as
         "Determine how elements should be grouped" or "Decide what conditions
         define a valid result."

    The purpose is to strengthen the student's problem-solving skills by guiding
    them through abstract reasoning and strategic planning without writing code.

    CONTENT RESTRICTIONS:
    - NO code/syntax
    - NO variables/functions
    - NO language features
    Format:
    1. [Step one]
    2. [Step two]
    3. [Step three]
    (continue up to 5 steps maximum if needed)

    Do not include any introduction or explanation — just the numbered list of
    steps.
    """
        try:
            response = await self._generate(
                prompt,
                system_message=(
                    "You are a computer science educator guiding students "
                    "through algorithmic thinking without code."
                ),
            )
            return self._parse_numbered_list(response)
        except (BaseException, Exception):
            return [
                "Read and understand the problem statement carefully.",
                "Identify the type of input and expected output.",
                "Determine what operations or transformations are needed.",
                "Break down the logic into sequential decision points.",
                "Think of edge cases or exceptions that may affect the result.",
            ]


    async def generate_pseudocode(self, code: str, problem: str) -> dict[str, str]:
        """Assist students in translating their code into structured pseudocode."""
        prompt = f"""Problem Statement:
    {problem.strip()}

    Student's Code:
    {code.strip()}

    Write clean and readable pseudocode that helps the student understand the
    structure of the algorithm without worrying about syntax or
    language-specific features.

    Your pseudocode must follow these guidelines:

    - High-Level Logic: Capture the key steps and structure of the algorithm
      — not line-by-line code or specific language syntax.

    - No Variables or Language Syntax: Avoid specific variable names, function
      definitions, or code formatting. Use general terms like
      "loop through each item", "store result", "check condition".

    - Step-by-Step Clarity: Break the logic into numbered or clearly separated
      steps that follow a natural progression.

    - Abstract Operations: Focus on what is being done, not how it is written
      in code (e.g., say "sort the list" instead of using sort()).

    - Modular Thinking: Group steps logically (e.g., input validation,
      processing, result aggregation).

    - Avoid Over-Specification: Keep the pseudocode language-agnostic, simple,
      and intuitive.

    - Maximum 10-12 steps unless absolutely necessary.

    - Avoid actual implementation; focus on logic
    Examples of tone and format:

    - Start pseudocode with: "Pseudocode:"

    - Use indented bullet points, numbered steps, or simple phrases like:

    - "Initialize a result container"

    - "Loop through each element in the input"

    - "Check if the current element meets the condition"

    - "Update the result if condition is satisfied"

    - "Return the final result"


    - Use formatting as shown:

    === PSEUDOCODE ===
    [Logical steps as pseudocode]
    === EXPLANATION ===
    [Explain the flow and reasoning behind the steps]

    Highlight any algorithms or common programming patterns being used
    (e.g., sorting, searching, recursion).
    """
        try:
            response = await self._generate(
                prompt,
                system_message=(
                    "You are an educator helping students understand how to express "
                    "code logic in plain terms. Focus on clarity, structure, and "
                    "educational value. Avoid language-specific details or direct "
                    "implementation."
                ),
            )

            pseudocode = self._extract_pseudocode_block(response)
            explanation = self._extract_pseudocode_explanation(response)
        except Exception:
            logger.exception("Question generation failed")
            return "What edge cases should you consider for this problem?"
        else:
            return {
                "pseudocode": pseudocode,
                "explanation": explanation
                or "This pseudocode summarizes your algorithm logic.",
            }


    # Helper methods
    def _get_fallback_hints(self) -> list[str]:
        return [
            "Think carefully about the problem requirements and constraints",
            "Consider what data structures might help organize the information",
            "Try breaking the problem into smaller subproblems",
        ]

    def _parse_unstructured_error_response(self, text: str, error: str) -> dict:
        """Parse error explanation from unstructured text."""
        return {
            "error_type": self._extract_error_type(error),
            "explanation": self._extract_first_paragraph(text),
            "suggested_fixes": self._extract_bullet_points(text),
            "relevant_line": self._find_relevant_line(error),
            "common_mistakes": self._extract_common_mistakes(text),
        }

    def _extract_common_mistakes(self, text: str) -> list[str]:
        """Extract common mistakes from error explanation."""
        mistakes = []
        for line in text.split("\n"):
            if "common mistake" in line.lower() or "frequent error" in line.lower():
                mistake = re.sub(r".*:[-\s]*", "", line, flags=re.IGNORECASE).strip()
                if mistake:
                    mistakes.append(mistake)
        return mistakes if mistakes else ["Not checking for edge cases"]

    def _extract_code_snippet(self, text: str) -> str:
        """Extract code snippet from response with better pattern matching."""
        code_block = re.search(r"```(?:[a-z]*\n)?(.*?)\n?```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        indented_block = re.search(
            r"^( {4,}|\t+)(.*?)(?=\n\s*\w)", text, re.DOTALL | re.MULTILINE,
        )
        if indented_block:
            return indented_block.group(0).strip()

        code_lines = [
            line.strip()
            for line in text.split("\n")
            if re.search(r"[{}();=<>+\-*/]", line)
        ]
        if code_lines:
            return "\n".join(code_lines)

        return "// No code snippet provided"

    def _extract_first_paragraph(self, text: str) -> str:
        """Extract the first coherent paragraph with better filtering."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for p in paragraphs:
            if not re.match(r"^(Explanation|Error|Fix|Note):?\s*$", p, re.IGNORECASE):
                return p
        return "No explanation available"

    def _extract_bullet_points(self, text: str) -> list[str]:
        """Extract bullet points or numbered items with better pattern matching."""
        points: list[str] = []
        for line in text.split("\n"):
            match = re.match(r"^[\-\*\d+\.]\s*(.+)$", line.strip())
            if match:
                point = match.group(1)
                if (
                    len(point.split()) > self.MIN_BULLET_POINT_WORDS
                    and not point.endswith(":")
                ):
                    points.append(point)
        return points if points else ["Review the code carefully for logical errors"]

    def _extract_error_type(self, error: str) -> str:
        """Enhanced error type detection with more patterns."""
        error_patterns = {
            "IndexError": r"index(?: out of range| too large)",
            "KeyError": r"key(?: not found| error)",
            "TypeError": r"type(?: mismatch| error)",
            "ValueError": r"value(?: error| invalid)",
            "SyntaxError": r"syntax(?: error| invalid)",
            "NameError": r"name .* is not defined",
            "AttributeError": r"attribute .* not found",
            "ImportError": r"import(?: error| .* not found)",
            "ZeroDivisionError": r"division by zero",
            "RuntimeError": r"runtime error",
        }

        for err_type, pattern in error_patterns.items():
            if re.search(pattern, error, re.IGNORECASE):
                return err_type
        return "RuntimeError"

    def _find_relevant_line(self, error: str) -> int | None:
        """Enhanced line number detection with more patterns."""
        line_patterns = [
            r"line\s+(\d+)",
            r"at line (\d+)",
            r"\(line (\d+)\)",
            r"on line[: ]*(\d+)",
        ]

        for pattern in line_patterns:
            match = re.search(pattern, error, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _estimate_complexity(self, code: str) -> dict[str, str]:
        """Enhanced complexity estimation with nested loop detection."""
        code = code.lower()

        nested_loop_pattern = (
            r"(for\s*.*\s*in\s*.*:|while\s*\(.*\):)"
            r"[\s\S]*?"
            r"(for\s*.*\s*in\s*.*:|while\s*\(.*\):)"
        )
        if re.search(nested_loop_pattern, code):
            return {"time": "O(n²)", "space": "O(1)"}

        recursive_call_pattern = r"def\s+\w+\(.*\):\s*.*\w+\(.*\)"
        if re.search(recursive_call_pattern, code):
            return {"time": "O(2^n)", "space": "O(n)"}

        single_loop_pattern = r"(for\s*.*\s*in\s*.*:|while\s*\(.*\):)"
        if re.search(single_loop_pattern, code):
            return {"time": "O(n)", "space": "O(1)"}

        return {"time": "O(1)", "space": "O(1)"}

    def _extract_complexity(self, text: str, section: str) -> dict[str, str]:
        """Extract complexity notation with more patterns."""
        section_match = re.search(
            fr"{section}:\s*(.*?)(?:\n|$)", text, re.IGNORECASE,
        )
        if section_match:
            complexity_match = re.search(r"\bO\([^)]+\)", section_match.group(1))
            if complexity_match:
                return {"time": complexity_match.group(0), "space": "O(1)"}

            informal_patterns = {
                "constant": "O(1)",
                "linear": "O(n)",
                "quadratic": "O(n²)",
                "logarithmic": "O(log n)",
                "exponential": "O(2^n)",
            }
            for term, complexity in informal_patterns.items():
                if term in section_match.group(1).lower():
                    return {"time": complexity, "space": "O(1)"}

        return {"time": "Unknown", "space": "Unknown"}


    def _extract_section(self, text: str, section: str) -> list[str]:
        """Extract bullet points from a specific section with better parsing."""
        section_text = ""

        # Find the section content with more flexible matching
        section_pattern = fr"(?:{section}|{section[:-1]})\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)"
        section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
        if section_match:
            section_text = section_match.group(1)

        # Extract bullet points with better filtering
        points: list[str] = []
        for raw_line in section_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            # Match various bullet point formats
            match = re.match(r"^[\-\*\d+\.]\s*(.+)$", line)
            if match:
                point = match.group(1)
                # Skip points that are too short or just headers
                if (
                    len(point.split()) > self.MIN_POINT_WORDS
                    and not point.endswith(":")
                ):
                    points.append(point)
            elif len(line.split()) > self.MIN_CONTINUATION_WORDS:
                # Consider standalone lines as points
                points.append(line)

        return points if points else [f"No specific {section.lower()} suggestions"]

    def _parse_numbered_list(self, text: str) -> list[str]:
        """Split the text into lines. Then, strip leading and trailing whitespace from each line."""
        lines: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if line:  # Only add non-empty lines
                lines.append(line)
        return lines

    def _extract_pseudocode_block(self, response: str) -> str:
        """Extract the pseudocode section with better pattern matching."""
        # Try explicit delimiters first
        block_match = re.search(
            r"=== PSEUDOCODE ===(.*?)=== EXPLANATION ===", response, re.DOTALL,
        )
        if block_match:
            pseudocode = block_match.group(1).strip()
            # Clean up any remaining markdown code blocks
            return re.sub(
                r"^```.*?\n|\n```$", "", pseudocode, flags=re.MULTILINE,
            ).strip()

        # Try BEGIN/END format
        begin_match = re.search(
            r"BEGIN PSEUDOCODE(.*?)END PSEUDOCODE",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if begin_match:
            return begin_match.group(1).strip()

        # Try to extract the most code-like section
        code_lines: list[str] = []
        in_code_block = False

        for line in response.split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or (
                re.search(r"\b(begin|end|if|else|for|while|return)\b", line.lower())
                and not stripped_line.endswith(":")
            ):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        # Fallback to the entire response cleaned up
        return re.sub(r"^\s*\d+\.\s*", "", response, flags=re.MULTILINE).strip()

    def _extract_pseudocode_explanation(self, response: str) -> str:
        """Extract the explanation section with better pattern matching."""
        # Try explicit delimiter first
        explanation_match = re.search(
            r"=== EXPLANATION ===(.*?)(?:\n===|$)", response, re.DOTALL,
        )
        if explanation_match:
            return explanation_match.group(1).strip()

        # Look for common explanation patterns
        explanation_keywords = [
            "explanation",
            "description",
            "breakdown",
            "approach",
            "logic",
            "steps",
        ]

        for keyword in explanation_keywords:
            pattern = fr"{keyword}:\s*(.*?)(?=\n\s*\w+\s*:|$)"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                # Remove any remaining bullet points or numbers
                return re.sub(r"^\s*[\d\-*]\s*", "", explanation, flags=re.MULTILINE)

        # Try to find a paragraph that explains the pseudocode
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        for p in paragraphs:
            if re.search(r"(pseudocode|algorithm|approach)", p, re.IGNORECASE) and len(
                p.split(),
            ) > self.MIN_EXPLANATION_WORDS:
                return p

        return "This pseudocode represents the algorithmic approach."
