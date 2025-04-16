document.addEventListener('DOMContentLoaded', () => {
    // URL parsing and element selection
    const pathSegments = window.location.pathname.split('/').filter(Boolean);
    if (pathSegments.length < 6) {
        console.error('Invalid URL structure');
        return;
    }
    
    const courseId = pathSegments[1]; // This determines the language
    const topicId = pathSegments[3];
    const problemId = pathSegments[5];
    
    // Determine language based on courseId
    const language = courseId.toLowerCase() === 'javascript' ? 'javascript' : 'python';
    console.log(`Detected language: ${language}`);
    
    // Get DOM elements
    const runBtn = document.getElementById('run-btn');
    const submitBtn = document.getElementById('submit-btn')
    const hintBtn = document.getElementById('hint-btn');
    const explainBtn = document.getElementById('explain-btn');
    const optimiseBtn = document.getElementById('optimize-btn');
    const pseudocodeBtn = document.getElementById('pseudocode-btn');
    const stepsBtn = document.getElementById('steps-btn');
    const codeEditor = document.getElementById('code-editor');
    const resultsContainer = document.getElementById('results-container');
    const assistanceOutput = document.getElementById('assistance-output');
    
    // Validate all required elements exist
    if (!runBtn || !hintBtn || !explainBtn || !optimiseBtn || !pseudocodeBtn || !stepsBtn || !codeEditor || !resultsContainer || !assistanceOutput || !submitBtn) {
        console.error('Missing required DOM elements');
        return;
    }

    // Helper function to escape HTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function setLoading(button, isLoading, text = '') {
        button.disabled = isLoading;
        button.innerHTML = isLoading ? '<span class="spinner"></span> ' + text : text;
    }


    // Run button click handler
    runBtn.addEventListener('click', async () => {
        setLoading(runBtn, true, 'Running...');
        resultsContainer.innerHTML = '<div class="loading">Executing your code...</div>';
        try {
            const apiUrl = `${problemId}/execute`;
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language
                }),
                credentials: 'same-origin'
            });
    
            const responseText = await response.text();
            let results;
            
            try {
                results = JSON.parse(responseText);
            } catch (e) {
                throw new Error(`Invalid JSON response: ${responseText}`);
            }
    
            if (!response.ok) {
                throw new Error(results.detail || `HTTP error! Status: ${response.status}`);
            }
    
            displayResults(results, language);
        } catch (error) {
            console.error('Error:', error);
            resultsContainer.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>${error.message}</p>
                    ${error.stack ? `<pre>${error.stack}</pre>` : ''}
                </div>
            `;
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Run Code';
        }
    });

    // Hint button click handler
    hintBtn.addEventListener('click', async () => {
        hintBtn.disabled = true;
        hintBtn.innerHTML = '<span class="spinner"></span> Loading...';
        assistanceOutput.innerHTML = '<div class="loading">Generating hints...</div>';
        
        try {
            const response = await fetch(`${problemId}/hints`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language
                }),
                credentials: 'same-origin'
            });
    
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server responded with ${response.status}: ${errorText}`);
            }
    
            const hintData = await response.json();
            
            if (hintData.hint_text && hintData.hint_text.trim() !== '') {
                assistanceOutput.innerHTML = `
                    <div class="hint-container">
                        <h3>Hint (${hintData.hint_level + 1}/${hintData.max_level + 1})</h3>
                        <div class="hint-content">${hintData.hint_text}</div>
                        ${hintData.hint_level < hintData.max_level ? 
                            '<button class="next-hint-btn">Next Hint</button>' : 
                            '<p class="no-more-hints">No more hints available</p>'}
                    </div>
                `;
                
                const nextHintBtn = assistanceOutput.querySelector('.next-hint-btn');
                if (nextHintBtn) {
                    nextHintBtn.addEventListener('click', async () => {
                        await fetchNextHint(hintData.hint_level + 1);
                    });
                }
            } else {
                assistanceOutput.innerHTML = '<div class="no-hints">No hints available for this problem</div>';
            }
            
        } catch (error) {
            console.error('Hint error:', error);
            assistanceOutput.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            hintBtn.disabled = false;
            hintBtn.textContent = 'Get Hint';
        }
    });
    submitBtn.addEventListener('click', async () => {
        setLoading(submitBtn, true, 'Submitting...');
        resultsContainer.innerHTML = '<div class="loading">Submitting your solution...</div>';
        
        try {
            // 1. Prepare the request body exactly as backend expects
            const requestBody = {
                code: codeEditor.value,
                language: language, // Must be 'python' or 'javascript'
                user_id: getCurrentUserId() // REQUIRED field
            };
    
            const response = await fetch(`/course/${courseId}/topic/${topicId}/problem/${problemId}/submit`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token') || ''}`
                },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language,
                    user_id: getCurrentUserId()
                })
            });
        
            // 3. Handle response
            if (!response.ok) {
                const errorData = await response.json();
                let errorMessage = 'Submission failed';
                
                // Handle FastAPI validation errors
                if (errorData.detail) {
                    if (Array.isArray(errorData.detail)) {
                        errorMessage = errorData.detail.map(err => 
                            `${err.loc?.join('.')}: ${err.msg}`
                        ).join('\n');
                    } else {
                        errorMessage = errorData.detail;
                    }
                }
                
                throw new Error(errorMessage);
            }
    
            // 4. Process successful response
            const result = await response.json();
            
            // Display results matching your backend's response model
            displaySubmissionResults(result);
            
        } catch (error) {
            // 5. Show error to user
            resultsContainer.innerHTML = `
                <div class="error">
                    <h3>Submission Error</h3>
                    <pre>${escapeHtml(error.message)}</pre>
                    <button onclick="location.reload()">Try Again</button>
                </div>
            `;
            console.error('Submission error:', error);
        } finally {
            setLoading(submitBtn, false, 'Submit');
        }
    });
    
    // Helper functions
    function getCurrentUserId() {
        // Check localStorage first
        const userId = localStorage.getItem('userId');
        
        // If using session-based auth, check cookies
        if (!userId) {
            const cookieValue = document.cookie
                .split('; ')
                .find(row => row.startsWith('user_id='))
                ?.split('=')[1];
            if (cookieValue) return cookieValue;
        }
        
        // If using token-based auth, decode from JWT
        if (!userId) {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    const payload = JSON.parse(atob(token.split('.')[1]));
                    return payload.sub || payload.userId || 'anonymous';
                } catch (e) {
                    console.warn('Failed to decode token:', e);
                }
            }
        }
        
        console.warn('No user ID found - using anonymous');
        return 'anonymous'; // Fallback
    }
    
    function displaySubmissionResults(data) {
        // Ensure we have valid result data
        if (!data || typeof data !== 'object') {
            resultsContainer.innerHTML = `
                <div class="error">
                    <h3>Invalid Response</h3>
                    <p>Received malformed data from server</p>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
            return;
        }
    
        let html = `
            <div class="submission-result ${data.result?.passed ? 'passed' : 'failed'}">
                <h3>Submission ${data.status === 'completed' ? 'Successful' : 'Failed'}</h3>
                <p>Score: ${data.result?.score ?? 0}%</p>
                <p>Status: ${data.result?.passed ? 'Passed' : 'Failed'}</p>
        `;
    
        // Add feedback if available
        if (data.result?.feedback) {
            html += `
                <div class="feedback">
                    <h4>Feedback:</h4>
                    <pre>${escapeHtml(data.result.feedback)}</pre>
                </div>
            `;
        }
    
        // Add test results if available
        if (data.result?.test_results && Array.isArray(data.result.test_results)) {
            html += '<div class="test-results"><h4>Test Results:</h4>';
            
            data.result.test_results.forEach((test, index) => {
                const testNumber = index + 1;
                html += `
                    <div class="test-case ${test.passed ? 'passed' : 'failed'}">
                        <h5>Test ${testNumber}: ${test.passed ? '✓ Passed' : '✗ Failed'}</h5>
                        ${test.input ? `<div class="test-io"><span>Input:</span> <pre>${escapeHtml(test.input)}</pre></div>` : ''}
                        <div class="test-io"><span>Expected:</span> <pre>${escapeHtml(test.expected)}</pre></div>
                        ${test.output ? `<div class="test-io"><span>Output:</span> <pre>${escapeHtml(test.output)}</pre></div>` : ''}
                        ${test.error ? `<div class="test-error"><span>Error:</span> <pre>${escapeHtml(test.error)}</pre></div>` : ''}
                    </div>
                `;
            });
            
            html += '</div>'; // Close test-results div
        }
    
        html += '</div>'; // Close submission-result div
        resultsContainer.innerHTML = html;
    }
    
    // Utility function to escape HTML
    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            unsafe = String(unsafe);
        }
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Explain Error button click handler
    explainBtn.addEventListener('click', async () => {
        try {
            explainBtn.disabled = true;
            explainBtn.innerHTML = '<span class="spinner"></span> Analyzing...';
            assistanceOutput.innerHTML = '<div class="loading">Analyzing code for errors...</div>';
        
            // Execute code first
            const executeResponse = await fetch(`${problemId}/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language
                })
            });
      
            const executeResult = await executeResponse.json();
            if (!executeResponse.ok) {
                throw new Error(executeResult.detail || 'Execution failed');
            }
      
            // Extract error
            const error = extractErrorFromResponse(executeResult, language);
            if (!error) {
                assistanceOutput.innerHTML = `
                    <div class="success">
                        <h3>No Errors Found</h3>
                        <p>✅ Code executed successfully</p>
                        ${executeResult.stdout ? `<pre>${executeResult.stdout}</pre>` : ''}
                    </div>
                `;
                return;
            }
      
            // Get explanation
            assistanceOutput.innerHTML = '<div class="loading">Generating explanation...</div>';
            const explanation = await getErrorExplanation(codeEditor.value, error, language);
            displayExplanation(assistanceOutput, error, explanation, language);
      
        } catch (err) {
            assistanceOutput.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>${err.message}</p>
                </div>
            `;
            console.error('Error:', err);
        } finally {
            explainBtn.disabled = false;
            explainBtn.textContent = 'Explain Error';
        }
    });

    // Optimize button click handler
    optimiseBtn.addEventListener('click', async () => {
        optimiseBtn.disabled = true;
        optimiseBtn.innerHTML = '<span class="spinner"></span> Optimizing...';
        assistanceOutput.innerHTML = '<div class="loading">Optimizing your code...</div>';
    
        try {
            const response = await fetch(`${problemId}/optimize`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language
                }),
                credentials: 'same-origin'
            });
    
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server responded with ${response.status}: ${errorText}`);
            }
    
            const optimisationData = await response.json();
    
            if (optimisationData.optimised_code) {
                assistanceOutput.innerHTML = `
                    <div class="optimisation-result">
                        <h3>Optimized Code</h3>
                        <div class="code-comparison">
                            <div class="original-code">
                                <h4>Original Code</h4>
                                <pre>${escapeHtml(codeEditor.value)}</pre>
                            </div>
                            <div class="optimised-code">
                                <h4>Optimized Version</h4>
                                <pre>${escapeHtml(optimisationData.optimised_code)}</pre>
                            </div>
                        </div>
                        ${optimisationData.improvements ? `
                            <div class="improvements">
                                <h4>Improvements Made:</h4>
                                <ul>
                                    ${optimisationData.improvements.map(imp => `<li>${imp}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                        <button id="apply-optimisation" class="apply-btn">Apply Changes</button>
                    </div>
                `;
    
                document.getElementById('apply-optimisation').addEventListener('click', () => {
                    codeEditor.value = optimisationData.optimised_code;
                    assistanceOutput.innerHTML = '<div class="success">Optimized code applied!</div>';
                });
            } else {
                assistanceOutput.innerHTML = `
                    <div class="no-optimisation">
                        <h3>No Optimizations Found</h3>
                        <p>Your code is already well optimized!</p>
                    </div>
                `;
            }
    
        } catch (error) {
            console.error('Optimisation error:', error);
            assistanceOutput.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            optimiseBtn.disabled = false;
            optimiseBtn.textContent = 'Optimize Code';
        }
    });
    
    // Pseudocode button click handler
    pseudocodeBtn.addEventListener('click', async () => {
        pseudocodeBtn.disabled = true;
        pseudocodeBtn.innerHTML = '<span class="spinner"></span> Generating...';
        assistanceOutput.innerHTML = '<div class="loading">Generating pseudocode...</div>';
        
        try {
            const response = await fetch(`${problemId}/pseudocode`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate pseudocode');
            }

            const pseudocodeData = await response.json();
            assistanceOutput.innerHTML = `
                <div class="pseudocode-result">
                    <h3>Pseudocode</h3>
                    <pre>${escapeHtml(pseudocodeData.pseudocode)}</pre>
                </div>
            `;
        } catch (error) {
            assistanceOutput.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            pseudocodeBtn.disabled = false;
            pseudocodeBtn.textContent = 'Get Pseudocode';
        }
    });

    // Steps button click handler
    stepsBtn.addEventListener('click', async () => {
        stepsBtn.disabled = true;
        stepsBtn.textContent = 'Loading Steps...';
        assistanceOutput.innerHTML = '<div class="loading-message">Generating conceptual steps...</div>';
        
        try {
            const response = await fetch(`${problemId}/steps`, {
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.message || `HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.steps && Array.isArray(data.steps) && data.steps.length > 0) {
                let stepsHTML = '<div class="steps-container"><h3>Conceptual Steps:</h3><ol>';
                data.steps.forEach(step => {
                    stepsHTML += `<li>${escapeHtml(step)}</li>`;
                });
                stepsHTML += '</ol></div>';
                assistanceOutput.innerHTML = stepsHTML;
            } else {
                assistanceOutput.innerHTML = '<div class="no-steps-message">No steps could be generated for this problem.</div>';
            }
        } catch (error) {
            console.error('Error fetching conceptual steps:', error);
            assistanceOutput.innerHTML = `<div class="error-message">Error loading steps: ${escapeHtml(error.message)}</div>`;
        } finally {
            stepsBtn.disabled = false;
            stepsBtn.textContent = 'Conceptual Steps';
        }
    });
    
    // Helper function to fetch next hint
    async function fetchNextHint(nextLevel) {
        try {
            assistanceOutput.innerHTML = '<div class="loading">Loading next hint...</div>';
            
            const response = await fetch(`${problemId}/hints`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code: codeEditor.value,
                    language: language,
                    current_hint_level: nextLevel - 1
                }),
                credentials: 'same-origin'
            });
    
            const hintData = await response.json();
            
            if (hintData.hint_text && hintData.hint_text.trim() !== '') {
                assistanceOutput.innerHTML = `
                    <div class="hint-container">
                        <h3>Hint (${hintData.hint_level + 1}/${hintData.max_level + 1})</h3>
                        <div class="hint-content">${hintData.hint_text}</div>
                        ${hintData.hint_level < hintData.max_level ? 
                            '<button class="next-hint-btn">Next Hint</button>' : 
                            '<p class="no-more-hints">No more hints available</p>'}
                    </div>
                `;
                
                const nextHintBtn = assistanceOutput.querySelector('.next-hint-btn');
                if (nextHintBtn) {
                    nextHintBtn.addEventListener('click', async () => {
                        await fetchNextHint(hintData.hint_level + 1);
                    });
                }
            }
        } catch (error) {
            console.error('Error fetching next hint:', error);
            assistanceOutput.innerHTML = `
                <div class="error">
                    <h3>Error</h3>
                    <p>Failed to load next hint: ${error.message}</p>
                </div>
            `;
        }
    }
    

    // Language-specific error extraction
    function extractErrorFromResponse(response, lang) {
        if (!response) return null;
      
        // Check visible test results first
        if (response.visible_results?.length) {
            const failedTest = response.visible_results.find(t => t.status === "failed" || t.passed === false);
            if (failedTest) {
                return failedTest.message || failedTest.error || failedTest.diff || "Test failed";
            }
        }
      
        // Check hidden tests
        if (response.hidden_passed === false) {
            return "Hidden tests failed - check your implementation";
        }
      
        // Language-specific error fields
        if (lang === 'javascript') {
            if (response.error && typeof response.error === 'object') {
                return `${response.error.name || 'Error'}: ${response.error.message}`;
            }
            if (response.stack) {
                return response.stack;
            }
        }
      
        // Common error fields
        const errorFields = ['error', 'stderr', 'message', 'detail', 'exception', 'traceback'];
        for (const field of errorFields) {
            if (response[field]) {
                return response[field];
            }
        }
      
        return null;
    }
      
    // Get error explanation from server
    async function getErrorExplanation(code, error, lang) {
        try {
            const response = await fetch(`${problemId}/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: code,
                    language: lang,
                    error: error
                })
            });
      
            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.detail || 'Failed to get explanation');
            }
      
            return await response.json();
        } catch (err) {
            console.error('Explanation error:', err);
            return {
                error_type: 'Analysis Error',
                explanation: 'Could not generate detailed explanation.',
                suggested_fixes: ['Check your code for syntax errors', 'Review the error message'],
                original_error: error
            };
        }
    }
      
    // Display error explanation
    function displayExplanation(container, error, explanation, lang) {
        let errorType = explanation.error_type || 'Error Analysis';
        
        // Special handling for JavaScript error types
        if (lang === 'javascript' && error.includes(':')) {
            errorType = error.split(':')[0].trim();
        }
        
        container.innerHTML = `
            <div class="error-explanation">
                <h3>${errorType}</h3>
                ${explanation.relevant_line ? `
                    <div class="relevant-line">
                        <strong>Relevant Line:</strong> ${explanation.relevant_line}
                    </div>
                ` : ''}
                <div class="explanation">
                    ${explanation.explanation || 'No detailed explanation available.'}
                </div>
                ${explanation.suggested_fixes?.length ? `
                    <h4>Suggested Fixes:</h4>
                    <ul class="fixes">
                        ${explanation.suggested_fixes.map(fix => `<li>${fix}</li>`).join('')}
                    </ul>
                ` : ''}
                <div class="original-error">
                    <h4>Original Error:</h4>
                    <pre>${error}</pre>
                </div>
            </div>
        `;
    }

    // Display results with language-specific handling
    function displayResults(results, lang) {
        console.log("Raw results from server:", results);
        
        if (!results) {
            resultsContainer.innerHTML = '<div class="error">No response from server</div>';
            return;
        }
    
        let html = '<h3>Execution Results</h3><div class="test-results">';
    
        // Handle execution errors
        if (results.error) {
            html += `
                <div class="error">
                    <h4>Execution Error</h4>
                    <pre>${results.error}</pre>
                    ${results.traceback ? `<pre class="traceback">${results.traceback}</pre>` : ''}
                </div>
            `;
            resultsContainer.innerHTML = html;
            return;
        }
    
        // Display visible test cases
        if (results.visible_results && results.visible_results.length > 0) {
            results.visible_results.forEach((test, index) => {
                html += `
                    <div class="test-case ${test.passed ? 'passed' : 'failed'}">
                        <h5>Test Case ${index + 1}: ${test.passed ? '✓ Passed' : '✗ Failed'}</h5>
                        ${test.input ? `<p><strong>Input:</strong> <code>${escapeHtml(test.input)}</code></p>` : ''}
                        <p><strong>Expected:</strong> <code>${escapeHtml(test.expected)}</code></p>
                        <p><strong>Output:</strong> <code>${escapeHtml(test.output)}</code></p>
                        ${test.error ? `<div class="error-details"><strong>Error:</strong><pre>${escapeHtml(test.error)}</pre></div>` : ''}
                    </div>
                `;
            });
        }
    
        // Add hidden tests summary
        if (results.hidden_passed !== undefined) {
            html += `
                <div class="hidden-summary ${results.hidden_passed ? 'passed' : 'failed'}">
                    <h4>Hidden Tests: ${results.hidden_passed ? '✓ Passed' : '✗ Failed'}</h4>
                    <p>${results.hidden_passed ? 
                        'All hidden tests passed successfully!' : 
                        'Some hidden tests did not pass.'}
                    </p>
                </div>
            `;
        }
    
        html += '</div>';
        resultsContainer.innerHTML = html;
    }
});