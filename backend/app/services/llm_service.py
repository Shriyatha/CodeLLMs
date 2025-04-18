from typing import List, Dict, Optional, Tuple, Any
import httpx
import re
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from configs.config import Config
import json
from enum import Enum
from app.services.mlflow_logger import MLFlowLogger

logger = MLFlowLogger()  

class HintLevel(Enum):
    GENERAL = 1
    DIRECTIONAL = 2
    SPECIFIC = 3

class LLMService:
    def __init__(self):
        self.ollama_base_url = Config.OLLAMA_HOST.rstrip('/')
        self.default_model = Config.DEFAULT_LLM
        self.timeout = httpx.Timeout(Config.LLM_TIMEOUT)
        self.client = httpx.AsyncClient()
        self.mlflow_logger = MLFlowLogger
        self.max_retries = 3

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate(self, prompt: str, system_message: Optional[str] = None, temperature: float = 0.3) -> str:
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
                    "stream": False
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            try:
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                if not content:
                    raise ValueError("Empty response content")
                return content
            except json.JSONDecodeError:
                text = response.text.strip()
                if text:
                    return text
                raise ValueError("Empty response")

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.log_llm_interaction(
                prompt={"prompt": prompt, "system_message": system_message},
                response=error_msg,
                metadata={"error": True, "status_code": e.response.status_code}
            )
            raise RuntimeError(f"LLM API error: {e.response.text[:200]}")
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.log_llm_interaction(
                prompt={"prompt": prompt, "system_message": system_message},
                response=error_msg,
                metadata={"error": True, "exception": str(e)}
            )
            raise RuntimeError("Could not process LLM response")

    async def get_progressive_hints(self, problem: str, code: str, current_level: int) -> Tuple[List[str], int]:
        """Generate tiered hints that gradually guide toward solution"""
        prompt = f"""Problem Statement:
{problem.strip()}

Student's Current Code:
{code.strip()}

Generate exactly 3 progressive hints to guide the student toward solving this problem:
1. First hint (Level 1): General direction without specifics
2. Second hint (Level 2): More specific guidance about approach
3. Third hint (Level 3): Concrete suggestion without full solution

Format your response EXACTLY like this:
1. [First hint text]
2. [Second hint text]
3. [Third hint text]"""

        try:
            response = await self._generate(
                prompt,
                system_message="You are a programming tutor. Provide only the 3 numbered hints in the specified format."
            )
            
            # Log successful interaction
            logger.log_llm_interaction(
                prompt={"problem": problem, "code": code, "current_level": current_level},
                response=response,
                metadata={"method": "get_progressive_hints"}
            )
            
            # Parse hints and ensure we have exactly 3
            hints = []
            for line in response.split('\n'):
                match = re.match(r'^\d+\.\s*(.+)$', line.strip())
                if match and len(hints) < 3:
                    hints.append(match.group(1))
            
            if len(hints) < 3:
                hints.extend(self._get_fallback_hints()[len(hints):3])
                
            return hints[:3], 3
            
        except Exception as e:
            error_msg = f"Hint generation failed: {str(e)}"
            logger.log_llm_interaction(
                prompt={"problem": problem, "code": code, "current_level": current_level},
                response=error_msg,
                metadata={"error": True, "exception": str(e)}
            )
            return self._get_fallback_hints(), 3

    async def explain_errors(
        self,
        error: str,
        code: str,
        problem: str,
        language: str = "python"
    ) -> Dict:
        """Generate structured error explanation with actionable fixes"""
        run_name = f"ExplainError_{language}"
        system_message = "You are a patient programming tutor explaining errors. Return valid JSON."

        if self.mlflow_logger:
            with self.mlflow_logger.start_run(run_name=run_name):
                self.mlflow_logger.log_param("language", language)
                self.mlflow_logger.log_param("problem", problem)
                self.mlflow_logger.log_text(code, "problematic_code.txt")
                self.mlflow_logger.log_text(error, "error_message.txt")

                prompt = f"""Programming Language: {language}
    Problem Description:
    {problem}

    Error Message:
    {error}

    Problematic Code:
    {code}

    Analyze this error and provide:
    1. Error type classification (e.g., "IndexError")
    2. Clear explanation of why it occurred
    3. 2-3 suggested fixes (bullet points)
    4. Relevant line number if apparent
    5. Common mistakes that lead to this error

    Format your response as JSON with these keys:
    {{
        "error_type": "",
        "explanation": "",
        "suggested_fixes": [],
        "relevant_line": null,
        "common_mistakes": []
    }}"""

                try:
                    response = await self._generate(
                        prompt,
                        system_message=system_message
                    )
                    # Log successful interaction
                    self.mlflow_logger.log_dict(
                        {"prompt": prompt, "system_message": system_message},
                        "llm_prompt.json"
                    )
                    self.mlflow_logger.log_text(response, "llm_response.txt")
                    self.mlflow_logger.set_tag("llm_interaction_status", "success")

                    try:
                        result = json.loads(response)
                        explanation_result = {
                            "error_type": result.get("error_type", self._extract_error_type(error)),
                            "explanation": result.get("explanation", ""),
                            "suggested_fixes": result.get("suggested_fixes", []),
                            "relevant_line": result.get("relevant_line", self._find_relevant_line(error)),
                            "common_mistakes": result.get("common_mistakes", [])
                        }
                        self.mlflow_logger.log_dict(explanation_result, "error_explanation.json")
                        return explanation_result
                    except json.JSONDecodeError as json_err:
                        unstructured_result = self._parse_unstructured_error_response(response, error)
                        self.mlflow_logger.log_dict(unstructured_result, "unstructured_error_explanation.json")
                        self.mlflow_logger.log_text(f"JSON Decode Error: {json_err}", "json_decode_error.txt")
                        return unstructured_result

                except Exception as e:
                    error_msg = f"Error explanation failed: {str(e)}"
                    self.mlflow_logger.log_text(error_msg, "error_explanation_failure.txt")
                    self.mlflow_logger.set_tag("llm_interaction_status", "failure")
                    explanation_result = {
                        "error_type": self._extract_error_type(error),
                        "explanation": f"Error occurred during explanation: {str(e)}",
                        "suggested_fixes": ["Review the error message and code"],
                        "relevant_line": self._find_relevant_line(error),
                        "common_mistakes": []
                    }
                    self.mlflow_logger.log_dict(explanation_result, "error_explanation_failure_details.json")
                    return explanation_result
                finally:
                    pass # 'with' statement handles end_run
        else:
            # Handle the case where mlflow_logger is None
            return {
                "error_type": self._extract_error_type(error),
                "explanation": "MLflow logging is not available.",
                "suggested_fixes": [],
                "relevant_line": self._find_relevant_line(error),
                "common_mistakes": []
            }
        
    async def analyze_optimizations(self, code: str, problem: str, language: str) -> Dict:
        prompt = f"""Analyze this {language} code and provide optimization suggestions.
        
Problem Description:
{problem.strip()}

Current Solution:
```{language.lower()}
{code.strip()}
Provide a detailed analysis including:

Current time and space complexity
Suggested optimizations
Readability improvements
Best practice recommendations
Important edge cases
Clear explanation
Relevant code snippet
Format your response as valid JSON with these exact keys:
{{
"current_complexity": {{"time": "", "space": ""}},
"suggested_complexity": {{"time": "", "space": ""}},
"optimization_suggestions": [],
"readability_suggestions": [],
"best_practice_suggestions": [],
"edge_cases": [],
"explanation": "",
"code_snippet": ""
}}"""
        try:
            response = await self._generate(
                prompt,
                system_message=f"""You are a senior {language} engineer reviewing code. 
                Return valid JSON with all requested fields."""
            )
            
            # Log successful interaction
            logger.log_llm_interaction(
                prompt={"code": code, "problem": problem, "language": language},
                response=response,
                metadata={"method": "analyze_optimizations"}
            )
            
            return self._parse_optimization_response(response, code, language)
                
        except Exception as e:
            error_msg = f"Optimization analysis failed: {str(e)}"
            logger.log_llm_interaction(
                prompt={"code": code, "problem": problem, "language": language},
                response=error_msg,
                metadata={"error": True, "exception": str(e)}
            )
            return self._parse_unstructured_optimization_response("", code)

    def _parse_optimization_response(self, response: str, code: str, language: str) -> Dict:
        default_response = {
            "current_complexity": self._estimate_complexity(code),
            "suggested_complexity": {"time": "", "space": ""},
            "optimization_suggestions": [],
            "readability_suggestions": [],
            "best_practice_suggestions": [],
            "edge_cases": [],
            "explanation": "",
            "code_snippet": ""
        }

        try:
            # Try to parse as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try extracting JSON from markdown
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    return self._parse_unstructured_optimization_response(response, code)

            # Validate and merge with defaults
            validated = default_response.copy()
            
            # Handle complexities
            if "current_complexity" in result:
                validated["current_complexity"] = {
                    "time": str(result["current_complexity"].get("time", "")),
                    "space": str(result["current_complexity"].get("space", ""))
                }
            
            if "suggested_complexity" in result:
                validated["suggested_complexity"] = {
                    "time": str(result["suggested_complexity"].get("time", "")),
                    "space": str(result["suggested_complexity"].get("space", ""))
                }

            # Handle list fields
            list_fields = [
                "optimization_suggestions",
                "readability_suggestions", 
                "best_practice_suggestions",
                "edge_cases"
            ]
            for field in list_fields:
                if field in result:
                    if isinstance(result[field], list):
                        validated[field] = [str(item) for item in result[field] if item]
                    elif result[field]:
                        validated[field] = [str(result[field])]

            # Handle string fields
            if "explanation" in result and result["explanation"]:
                validated["explanation"] = str(result["explanation"])
            
            if "code_snippet" in result and result["code_snippet"]:
                validated["code_snippet"] = str(result["code_snippet"])
            else:
                validated["code_snippet"] = self._extract_code_snippet(response)

            return validated

        except Exception as e:
            logger.warning(f"Optimization response parsing failed: {str(e)}")
            return self._parse_unstructured_optimization_response(response, code)

    def _parse_unstructured_optimization_response(self, text: str, code: str) -> Dict:
        return {
            "current_complexity": self._extract_complexity(text, "Current") or self._estimate_complexity(code),
            "suggested_complexity": self._extract_complexity(text, "Suggested") or {"time": "", "space": ""},
            "optimization_suggestions": self._extract_section(text, "Optimization"),
            "readability_suggestions": self._extract_section(text, "Readability"),
            "best_practice_suggestions": self._extract_section(text, "Best Practice"),
            "edge_cases": self._extract_section(text, "Edge Case"),
            "explanation": self._extract_first_paragraph(text),
            "code_snippet": self._extract_code_snippet(text)
        }

    async def generate_conceptual_steps(self, problem: str) -> List[str]:
        """Break a coding problem into logical, high-level conceptual steps."""
        prompt = f"""You are helping a student solve this programming problem:

    {problem.strip()}

    Break it into 3 to 5 **conceptual steps** — high-level thinking steps needed before writing code.
    Each step should:
    - Be one sentence long
    - Build logically on the previous
    - Avoid any code, variable names, or syntax
    - Focus only on thought process and planning

    Format:
    1. [Step one]
    2. [Step two]
    3. [Step three]
    (continue up to 5 steps maximum if needed)

    Do not include any introduction or explanation — just the numbered list of steps.
    """
        try:
            response = await self._generate(
                prompt,
                system_message="You are a computer science educator guiding students through algorithmic thinking without code."
            )
            return self._parse_numbered_list(response)
        except Exception as e:
            logger.error(f"Step generation failed: {str(e)}")
            return [
                "Read and understand the problem statement carefully.",
                "Identify the type of input and expected output.",
                "Determine what operations or transformations are needed.",
                "Break down the logic into sequential decision points.",
                "Think of edge cases or exceptions that may affect the result."
            ]

    async def ask_targeted_question(self, problem: str, code: str, focus_area: str) -> str:
        """Generate a Socratic question to guide thinking"""
        prompt = f"""Problem: {problem}
Current code:
{code}
Generate one targeted question about {focus_area} that will help the student:

Discover the solution themselves
Consider important aspects they may have missed
Think more deeply about the problem
The question should:

Be open-ended
Relate specifically to their code
Not reveal the answer
Be concise (1 sentence)"""
        try:
            return await self._generate(
                prompt,
                system_message="You are a Socratic tutor asking guiding questions.",
                temperature=0.7  # Slightly more creative
            )
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return "What edge cases should you consider for this problem?"
        
    async def generate_pseudocode(self, code: str, problem: str) -> Dict[str, str]:
        """
        Assist students in translating their code into structured, language-agnostic pseudocode.
        The focus is on helping understand logic and algorithm flow, not direct implementation.
        """
        prompt = f"""Problem Statement:
    {problem.strip()}

    Student's Code:
    {code.strip()}

    Convert the above code into clear, educational pseudocode that follows these rules:

    - Use BEGIN/END blocks to organize logic
    - Explain each step with concise comments
    - Use simple, language-independent logic
    - Do NOT include language-specific syntax
    - Avoid actual implementation; focus on logic
    - Use formatting as shown:

    === PSEUDOCODE ===
    [Logical steps as pseudocode]
    === EXPLANATION ===
    [Explain the flow and reasoning behind the steps]

    Highlight any algorithms or common programming patterns being used (e.g., sorting, searching, recursion).
    """
        try:
            response = await self._generate(
                prompt,
                system_message=(
                    "You are an educator helping students understand how to express code logic in plain terms. "
                    "Focus on clarity, structure, and educational value. Avoid language-specific details or direct implementation."
                )
            )

            pseudocode = self._extract_pseudocode_block(response)
            explanation = self._extract_pseudocode_explanation(response)

            return {
                "pseudocode": pseudocode,
                "explanation": explanation or "This pseudocode summarizes your algorithm logic using structured plain-language steps."
            }

        except Exception as e:
            logger.error(f"Pseudocode generation failed: {str(e)}", exc_info=True)
            return {
                "pseudocode": "Could not generate pseudocode.",
                "explanation": f"Error: {str(e)}. Please review your code or simplify it for better processing."
            }

        
    

    # Helper methods
    def _get_fallback_hints(self) -> List[str]:
        return [
            "Think carefully about the problem requirements and constraints",
            "Consider what data structures might help organize the information",
            "Try breaking the problem into smaller subproblems you can solve individually"
        ]

    def _parse_unstructured_error_response(self, text: str, error: str) -> Dict:
        """Parse error explanation from unstructured text"""
        return {
            "error_type": self._extract_error_type(error),
            "explanation": self._extract_first_paragraph(text),
            "suggested_fixes": self._extract_bullet_points(text),
            "relevant_line": self._find_relevant_line(error),
            "common_mistakes": self._extract_common_mistakes(text)
        }    

    def _extract_common_mistakes(self, text: str) -> List[str]:
        """Extract common mistakes from error explanation"""
        mistakes = []
        for line in text.split('\n'):
            if "common mistake" in line.lower() or "frequent error" in line.lower():
                mistake = re.sub(r'.*:[-\s]*', '', line, flags=re.IGNORECASE).strip()
                if mistake:
                    mistakes.append(mistake)
        return mistakes if mistakes else ["Not checking for edge cases"]

    def _extract_code_snippet(self, text: str) -> str:
        """Extract code snippet from response with better pattern matching"""
        # Try explicit code block markers first
        code_block = re.search(r'```(?:[a-z]*\n)?(.*?)\n?```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        # Then look for indented blocks that look like code
        indented_block = re.search(r'^( {4,}|\t+)(.*?)(?=\n\s*\w)', text, re.DOTALL | re.MULTILINE)
        if indented_block:
            return indented_block.group(0).strip()
        
        # Finally look for lines that look like code
        code_lines = []
        for line in text.split('\n'):
            if re.search(r'[{}();=<>+\-*/]', line):  # Looks like code
                code_lines.append(line.strip())
        if code_lines:
            return '\n'.join(code_lines)
        
        return "// No code snippet provided"

    def _extract_first_paragraph(self, text: str) -> str:
        """Extract the first coherent paragraph with better filtering"""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        for p in paragraphs:
            # Skip lines that are just section headers
            if not re.match(r'^(Explanation|Error|Fix|Note):?\s*$', p, re.IGNORECASE):
                return p
        return "No explanation available"

    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points or numbered items with better pattern matching"""
        points = []
        for line in text.split('\n'):
            # Match both bullet and numbered points
            match = re.match(r'^[\-\*\d+\.]\s*(.+)$', line.strip())
            if match:
                point = match.group(1)
                # Skip points that are too short or just headers
                if len(point.split()) > 2 and not point.endswith(':'):
                    points.append(point)
        return points if points else ["Review the code carefully for logical errors"]

    def _extract_error_type(self, error: str) -> str:
        """Enhanced error type detection with more patterns"""
        error_patterns = {
            'IndexError': r'index(?: out of range| too large)',
            'KeyError': r'key(?: not found| error)',
            'TypeError': r'type(?: mismatch| error)',
            'ValueError': r'value(?: error| invalid)',
            'SyntaxError': r'syntax(?: error| invalid)',
            'NameError': r'name .* is not defined',
            'AttributeError': r'attribute .* not found',
            'ImportError': r'import(?: error| .* not found)',
            'ZeroDivisionError': r'division by zero',
            'RuntimeError': r'runtime error'
        }
        
        for err_type, pattern in error_patterns.items():
            if re.search(pattern, error, re.IGNORECASE):
                return err_type
        return 'RuntimeError'

    def _find_relevant_line(self, error: str) -> Optional[int]:
        """Enhanced line number detection with more patterns"""
        line_patterns = [
            r'line\s+(\d+)',
            r'at line (\d+)',
            r'\(line (\d+)\)',
            r'on line[: ]*(\d+)'
        ]
        
        for pattern in line_patterns:
            match = re.search(pattern, error, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    def _estimate_complexity(self, code: str) -> Dict[str, str]:
        """Enhanced complexity estimation with nested loop detection"""
        code = code.lower()
        
        # Check for nested loops
        nested_loop_pattern = r'(for\s*.*\s*in\s*.*:|while\s*\(.*\):)[\s\S]*?(for\s*.*\s*in\s*.*:|while\s*\(.*\):)'
        if re.search(nested_loop_pattern, code):
            return {"time": "O(n²)", "space": "O(1)"}
        
        # Check for recursion
        if re.search(r'def\s+\w+\(.*\):\s*.*\w+\(.*\)', code):
            return {"time": "O(2^n)", "space": "O(n)"}  # Basic assumption for recursion
            
        # Check for single loops
        if re.search(r'(for\s*.*\s*in\s*.*:|while\s*\(.*\):)', code):
            return {"time": "O(n)", "space": "O(1)"}
            
        return {"time": "O(1)", "space": "O(1)"}

    def _extract_complexity(self, text: str, section: str) -> Dict[str, str]:
        """Extract complexity notation with more patterns"""
        section_match = re.search(fr'{section}:\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
        if section_match:
            complexity_match = re.search(r'\bO\([^)]+\)', section_match.group(1))
            if complexity_match:
                return {"time": complexity_match.group(0), "space": "O(1)"}
            
            # Look for informal complexity descriptions
            informal_patterns = {
                'constant': 'O(1)',
                'linear': 'O(n)',
                'quadratic': 'O(n²)',
                'logarithmic': 'O(log n)',
                'exponential': 'O(2^n)'
            }
            for term, complexity in informal_patterns.items():
                if term in section_match.group(1).lower():
                    return {"time": complexity, "space": "O(1)"}
                    
        return {"time": "Unknown", "space": "Unknown"}

    def _extract_section(self, text: str, section: str) -> List[str]:
        """Extract bullet points from a specific section with better parsing"""
        section_text = ""
        
        # Find the section content with more flexible matching
        section_pattern = fr'(?:{section}|{section[:-1]})\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)'
        section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
        if section_match:
            section_text = section_match.group(1)
        
        # Extract bullet points with better filtering
        points = []
        for line in section_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Match various bullet point formats
            match = re.match(r'^[\-\*\d+\.]\s*(.+)$', line)
            if match:
                point = match.group(1)
                # Skip points that are too short or just headers
                if len(point.split()) > 2 and not point.endswith(':'):
                    points.append(point)
            elif len(line.split()) > 3:  # Consider standalone lines as points
                points.append(line)
                
        return points if points else [f"No specific {section.lower()} suggestions"]

    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered steps from response with better validation"""
        steps = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Match numbered items more flexibly
            match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
            if match:
                step = match.group(1)
                # Skip steps that are too short or just headers
                if len(step.split()) > 2:
                    steps.append(step)
            elif len(steps) > 0 and len(line.split()) > 3:  # Continuation lines
                steps[-1] += " " + line
                
        return steps[:5]  # Return max 5 quality steps

    def _extract_pseudocode_block(self, response: str) -> str:
        """Extract the pseudocode section with better pattern matching"""
        # Try explicit delimiters first
        block_match = re.search(r'=== PSEUDOCODE ===(.*?)=== EXPLANATION ===', response, re.DOTALL)
        if block_match:
            pseudocode = block_match.group(1).strip()
            # Clean up any remaining markdown code blocks
            return re.sub(r'^```.*?\n|\n```$', '', pseudocode, flags=re.MULTILINE).strip()
        
        # Try BEGIN/END format
        begin_match = re.search(r'BEGIN PSEUDOCODE(.*?)END PSEUDOCODE', response, re.DOTALL | re.IGNORECASE)
        if begin_match:
            return begin_match.group(1).strip()
        
        # Try to extract the most code-like section
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block or (re.search(r'\b(begin|end|if|else|for|while|return)\b', line.lower()) and not line.strip().endswith(':')):
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback to the entire response cleaned up
        return re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE).strip()

    def _extract_pseudocode_explanation(self, response: str) -> str:
        """Extract the explanation section with better pattern matching"""
        # Try explicit delimiter first
        explanation_match = re.search(r'=== EXPLANATION ===(.*?)(?:\n===|$)', response, re.DOTALL)
        if explanation_match:
            return explanation_match.group(1).strip()
        
        # Look for common explanation patterns
        explanation_keywords = [
            'explanation', 'description', 'breakdown', 
            'approach', 'logic', 'steps'
        ]
        
        for keyword in explanation_keywords:
            pattern = fr'{keyword}:\s*(.*?)(?=\n\s*\w+\s*:|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                # Remove any remaining bullet points or numbers
                explanation = re.sub(r'^\s*[\d\-*]\s*', '', explanation, flags=re.MULTILINE)
                return explanation
                
        # Try to find a paragraph that explains the pseudocode
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        for p in paragraphs:
            if re.search(r'(pseudocode|algorithm|approach)', p, re.IGNORECASE):
                if len(p.split()) > 15:  # Only return longer explanations
                    return p
                    
        return "This pseudocode represents the algorithmic approach to solve the problem."