"""Code execution service for running code in isolated Docker containers."""

import asyncio
import contextlib
import io
import logging
import tarfile

import docker
import mlflow
from app.services.mlflow_logger import MLFlowLogger
from docker.models.containers import Container


class CodeExecutionService:
    """Service for executing code in isolated Docker containers with result tracking."""

    def __init__(self) -> None:
        """Initialize the code execution service with Docker client and configuration."""
        self.client = docker.from_env()
        self.timeout = 10  # Increased timeout to 10 seconds
        self.mem_limit = "200m"  # Increased memory limit to 200MB
        self.cpu_period = 100000
        self.cpu_quota = 50000
        self.logger = logging.getLogger(__name__)
        self.max_output_size = 1024 * 1024  # 1MB max output
        self.mlflow_logger = MLFlowLogger()

    async def execute_code(self, code: str, language: str, test_cases: list[dict]) -> list[dict]:
        """Execute code with test cases in isolated Docker containers.

        Args:
            code: Source code to execute
            language: Programming language of the code
            test_cases: List of test cases to run the code against

        Returns:
            List of test case results with execution details

        """
        results = []
        run_execute = None

        try:
            run_execute = self.mlflow_logger.start_run(run_name=f"ExecuteCode_{language}")
            self.mlflow_logger.log_param("language", language)
            self.mlflow_logger.log_param("num_test_cases", len(test_cases))
            self.mlflow_logger.log_text(code, f"submitted_code_{language}.txt")

            # Determine container settings
            if language.lower() == "python":
                image = "python:3.12-slim"
                filename = "code.py"
                exec_cmd = ["python", f"/app/{filename}"]
            elif language.lower() == "javascript":
                image = "node:18-slim"
                filename = "code.js"
                exec_cmd = ["node", f"/app/{filename}"]
            else:
                error_msg = f"Unsupported language: {language}"
                logging.error("%s", error_msg)
            for i, case in enumerate(test_cases, 1):
                input_data = str(case.get("input", ""))
                expected = str(case.get("expected_output", ""))
                visible = case.get("visible", True)
                run_case = None

                try:
                    run_case = self.mlflow_logger.start_run(run_name=f"TestCase_{i}", nested=True)
                    self.mlflow_logger.log_param("test_case", i)
                    self.mlflow_logger.log_param("input", input_data)
                    self.mlflow_logger.log_param("expected_output", expected)
                    self.mlflow_logger.log_param("visible", visible)

                    container = None
                    try:
                        # Create and start container
                        container = await self._create_and_start_container(image)

                        # Write code to container
                        await self._write_code_to_container(container, code, filename)

                        # Execute code
                        exit_code, output = await self._execute_in_container(
                            container, exec_cmd, input_data,
                        )

                        # Process results
                        if len(output) > self.max_output_size:
                            output = output[:self.max_output_size] + "... [OUTPUT TRUNCATED]"

                        result = {
                            "test_case": i,
                            "input": input_data,
                            "output": output.strip(),
                            "expected": expected,
                            "passed": output.strip() == expected,
                            "visible": visible,
                            "error": None,
                            "exit_code": exit_code,
                        }
                        results.append(result)
                        self.mlflow_logger.log_dict(result, f"test_case_{i}_result.json")

                    except TimeoutError:
                        error_msg = f"Execution timed out after {self.timeout} seconds"
                        result = {
                            "test_case": i,
                            "input": input_data,
                            "output": "",
                            "expected": expected,
                            "passed": False,
                            "visible": visible,
                            "error": error_msg,
                            "exit_code": -1,
                        }
                        results.append(result)
                        self.mlflow_logger.log_dict(result, f"test_case_{i}_result.json")
                        self.mlflow_logger.log_text(error_msg, f"test_case_{i}_error.txt")

                    except (BaseException, Exception) as e:  # pylint: disable=broad-except
                        error_msg = str(e)
                        result = {
                            "test_case": i,
                            "input": input_data,
                            "output": "",
                            "expected": expected,
                            "passed": False,
                            "visible": visible,
                            "error": error_msg,
                            "exit_code": -1,
                        }
                        results.append(result)
                        self.mlflow_logger.log_dict(result, f"test_case_{i}_result.json")
                        self.mlflow_logger.log_text(error_msg, f"test_case_{i}_error.txt")

                    finally:
                        if container:
                            await self._cleanup_container(container)
                finally:
                    if run_case:
                        mlflow.end_run()

        except Exception as e:  # pylint: disable=broad-except
            self.mlflow_logger.log_text(f"Execution failed: {e!s}", "execution_failure.txt")
            error_msg = f"Code execution failed: {e!s}"
            raise RuntimeError(error_msg) from e
        finally:
            if run_execute:
                mlflow.end_run()

        return results

    async def _create_and_start_container(self, image: str) -> Container:
        """Create and start a container that stays running."""
        container = self.client.containers.create(
            image,
            command=["tail", "-f", "/dev/null"],  # Keeps container running
            stdin_open=True,
            tty=False,
            mem_limit=self.mem_limit,
            cpu_period=self.cpu_period,
            cpu_quota=self.cpu_quota,
            network_mode="none",
            working_dir="/app",  # Changed from /tmp for security
            detach=True,
        )

        container.start()

        # Wait for container to be ready
        for _ in range(5):
            container.reload()
            if container.status == "running":
                return container
            await asyncio.sleep(0.5)

        error_msg = f"Container failed to start. Status: {container.status}"
        raise RuntimeError(error_msg)


    async def _write_code_to_container(self, container: Container, code: str, filename: str) -> None:
        """Write code to container with verification."""
        # Create tar archive
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            file_data = code.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(file_data)
            tarinfo.mode = 0o444  # Read-only
            tar.addfile(tarinfo, io.BytesIO(file_data))

        tar_stream.seek(0)

        # Write to container
        container_path = "/app"  # Changed from /tmp for security
        if not container.put_archive(container_path, tar_stream):
            error_msg = "Failed to write code to container"
            raise RuntimeError(error_msg)

        # Verify file exists
        exit_code, _ = container.exec_run(["test", "-f", f"{container_path}/{filename}"])
        if exit_code != 0:
            error_msg = "Code file not found in container after writing"
            raise RuntimeError(error_msg)

    async def _execute_in_container(
        self, container: Container, exec_cmd: list[str], input_data: str,
    ) -> tuple[int, str]:
        """Execute command in container with input and return exit code and cleaned output."""
        # Step 1: Create the exec instance with stdin enabled
        exec_id = self.client.api.exec_create(
            container.id,
            cmd=exec_cmd,
            stdin=True,
            stdout=True,
            stderr=True,
            tty=False,
        )

        # Step 2: Start exec and open socket
        sock = self.client.api.exec_start(
            exec_id=exec_id["Id"],
            socket=True,
        )
        # Using protected member is unavoidable with the Docker API
        sock._sock.settimeout(self.timeout)  # pylint: disable=protected-access

        try:
            # Step 3: Send input if any
            if input_data:
                # Using protected member is unavoidable with the Docker API
                sock._sock.sendall((input_data + "\n").encode("utf-8"))  # pylint: disable=protected-access

            # Step 4: Receive and collect raw output
            raw_output = b""
            while True:
                try:
                    # Using protected member is unavoidable with the Docker API
                    chunk = sock._sock.recv(4096)  # pylint: disable=protected-access
                    if not chunk:
                        break
                    raw_output += chunk
                except TimeoutError:
                    break

        finally:
            # Step 5: Always close socket
            sock.close()

        # Step 6: Demux Docker's multiplexed output (stdout/stderr)
        def demux_output(stream: bytes) -> bytes:
            output = b""
            # Define the constant for header size
            header_size = 8

            while len(stream) > header_size:
                header = stream[:header_size]
                length = int.from_bytes(header[4:], byteorder="big")
                content = stream[header_size:header_size + length]
                output += content
                stream = stream[header_size + length:]
            return output

        cleaned_output = demux_output(raw_output).decode("utf-8", errors="replace")

        # Step 7: Get exit code
        exec_result = self.client.api.exec_inspect(exec_id["Id"])
        exit_code = exec_result["ExitCode"]

        return exit_code, cleaned_output

    async def _cleanup_container(self, container: Container) -> None:
        """Stop and remove container with error handling."""
        try:
            container.stop(timeout=1)
            container.remove(v=True, force=True)
        except (BaseException, Exception):  # pylint: disable=broad-except
            with contextlib.suppress(Exception):
                container.remove(v=True, force=True)

    async def compile_code(self, code: str, language: str) -> dict:
        """Check code syntax in a disposable container."""
        run_compile = None
        try:
            run_compile = self.mlflow_logger.start_run(run_name=f"CompileCode_{language}")
            self.mlflow_logger.log_param("language", language)
            self.mlflow_logger.log_text(code, f"compilation_code_{language}.txt")

            if language.lower() == "python":
                result = await self._compile_python(code)
            elif language.lower() == "javascript":
                result = await self._compile_javascript(code)
            else:
                error_msg = f"Unsupported language: {language}"
                logging.error("Unsupported language: %s", error_msg)
            self.mlflow_logger.log_dict(result, "compilation_result.json")

        except (BaseException, Exception) as e:  # pylint: disable=broad-except
            error_dict = {
                "success": False,
                "error": str(e),
                "raw_logs": "",
            }
            self.mlflow_logger.log_dict(error_dict, "compilation_error.json")
            return error_dict
        else:
            return result
        finally:
            if run_compile:
                mlflow.end_run()

    async def _compile_python(self, code: str) -> dict:
        """Python compilation check."""
        container = None
        try:
            # Create container that stays running
            container = self.client.containers.create(
                "python:3.12-slim",
                command=["tail", "-f", "/dev/null"],  # Keeps container running
                stdin_open=True,
                mem_limit=self.mem_limit,
                network_mode="none",
                working_dir="/app",  # Changed from /tmp for security
                detach=True,
            )
            container.start()

            # Wait for container to be ready
            for _ in range(5):
                container.reload()
                if container.status == "running":
                    break
                await asyncio.sleep(0.5)
            else:
                error_msg = "Container failed to start"
                raise RuntimeError(error_msg)

            # Write code to container
            await self._write_code_to_container(container, code, "code.py")

            # Create a properly formatted Python script
            compile_script = """\
import sys
try:
    with open("/app/code.py") as f:
        compile(f.read(), "code.py", "exec")
    sys.exit(0)
except SyntaxError as e:
    print(f"Line {e.lineno}: {e.msg}", file=sys.stderr)
    if e.text:
        print(e.text.strip(), file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
"""
            # Write the compile script to a temporary file in the container
            await self._write_code_to_container(container, compile_script, "compile_check.py")

            # Execute compilation check
            exit_code, output = await self._execute_in_container(
                container,
                ["python", "/app/compile_check.py"],
                "",
            )

            return {
                "success": exit_code == 0,
                "message": "Compilation successful" if exit_code == 0 else "Compilation failed",
                "error": output if exit_code != 0 else None,
                "raw_logs": output,
            }
        finally:
            if container:
                await self._cleanup_container(container)

    async def _compile_javascript(self, code: str) -> dict:
        """JavaScript compilation check."""
        container = None
        try:
            # Create container that stays running
            container = self.client.containers.create(
                "node:18-slim",
                command=["tail", "-f", "/dev/null"],  # Keeps container running
                stdin_open=True,
                mem_limit=self.mem_limit,
                network_mode="none",
                working_dir="/app",  # Changed from /tmp for security
                detach=True,
            )
            container.start()

            # Wait for container to be ready
            for _ in range(5):
                container.reload()
                if container.status == "running":
                    break
                await asyncio.sleep(0.5)
            else:
                error_msg = "Container failed to start"
                raise RuntimeError(error_msg)

            # Write code to container
            await self._write_code_to_container(container, code, "code.js")

            # Execute compilation check
            exit_code, output = await self._execute_in_container(
                container,
                ["node", "--check", "/app/code.js"],
                "",
            )

            return {
                "success": exit_code == 0,
                "message": "Compilation successful" if exit_code == 0 else "Compilation failed",
                "error": output if exit_code != 0 else None,
                "raw_logs": output,
            }
        finally:
            if container:
                await self._cleanup_container(container)
