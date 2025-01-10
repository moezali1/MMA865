if (chrome.sidePanel) {
    chrome.sidePanel
        .setPanelBehavior({ openPanelOnActionClick: true })
        .catch((error) => console.error(error));
}

// Handle clicking the extension icon as fallback
chrome.action.onClicked.addListener((tab) => {
    if (!chrome.sidePanel) {
        chrome.windows.create({
            url: 'sidebar.html',
            type: 'popup',
            width: 400,
            height: 600
        });
    }
});