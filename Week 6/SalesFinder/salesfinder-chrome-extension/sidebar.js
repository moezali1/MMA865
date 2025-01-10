document.getElementById('draftEmail').addEventListener('click', async () => {
    const resultContainer = document.getElementById('result');
    const spinner = document.getElementById('spinner');
    const emailContent = document.getElementById('emailContent');

    // Show spinner and container
    resultContainer.classList.remove('hidden');
    spinner.classList.remove('hidden');
    emailContent.innerHTML = '';

    try {
        // Get the active tab's content
        const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });

        if (!tab) {
            throw new Error('No active tab found');
        }

        const [{ result }] = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            function: () => {
                const selectedText = window.getSelection().toString();
                return selectedText || document.body.innerText;
            },
        });

        // Make API call
        const response = await fetch('http://localhost:8000/generate-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: result }),
        });

        if (!response.ok) {
            throw new Error(`API responded with status: ${response.status}`);
        }

        const data = await response.json();
        spinner.classList.add('hidden');

        // Handle both formats and format the markdown
        let textContent = data.text || data;

        // Pre-process the text to fix formatting issues
        textContent = textContent
            // Fix paragraph spacing
            .replace(/([^:\n])\n([A-Z])/g, '$1\n\n$2')
            // Make Current/Optimized bold and fix formatting
            .replace(/(\n[*-] )(Current: )/g, '$1**Current**: ')
            .replace(/(\n[*-] )(Optimized: )/g, '$1**Optimized**: ')
            // Remove extra line breaks after Current/Optimized
            .replace(/(\*\*Current\*\*:.*?)\n+(?=\*\*Optimized\*\*)/g, '$1\n')
            .replace(/(\*\*Optimized\*\*:.*?)\n+(?=[A-Z])/g, '$1\n')
            // Fix bullet points
            .replace(/^- /gm, '* ')
            // Fix nested list indentation
            .replace(/(\n\s*)- /g, '$1* ')
            // Remove triple line breaks
            .replace(/\n{3,}/g, '\n\n')
            // Clean up extra spaces
            .replace(/[ ]+$/gm, '')
            // Ensure proper paragraph breaks
            .trim();

        // Use marked with specific options
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false,
                smartLists: true
            });
            emailContent.innerHTML = marked.parse(textContent);
        } else {
            emailContent.textContent = textContent;
        }

        // Ensure visibility
        resultContainer.classList.remove('hidden');
        emailContent.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        spinner.classList.add('hidden');
        emailContent.textContent = 'Error: ' + error.message;
    }

    // Add copy button functionality
    const copyButton = document.getElementById('copyButton');
    copyButton.addEventListener('click', async () => {
        try {
            const content = emailContent.innerText;
            await navigator.clipboard.writeText(content);

            // Visual feedback
            copyButton.classList.add('copied');
            copyButton.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 6L9 17l-5-5"></path>
                </svg>
                Copied!
            `;

            // Reset after 2 seconds
            setTimeout(() => {
                copyButton.classList.remove('copied');
                copyButton.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                    Copy
                `;
            }, 2000);
        } catch (err) {
            console.error('Failed to copy text:', err);
        }
    });
});