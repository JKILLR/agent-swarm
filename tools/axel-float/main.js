const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const path = require('path');

let mainWindow = null;
let isVisible = true;

function createWindow() {
  const { width: screenWidth, height: screenHeight } = screen.getPrimaryDisplay().workAreaSize;

  // Small floating window in bottom-right corner
  const windowWidth = 380;
  const windowHeight = 500;

  mainWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x: screenWidth - windowWidth - 20,
    y: screenHeight - windowHeight - 20,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: true,
    skipTaskbar: true,
    hasShadow: false,  // Disable shadow to prevent ghost layer on drag
    // Removed vibrancy to prevent rendering artifacts during drag
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');

  // Allow window to be dragged
  mainWindow.setMovable(true);

  // Keep on top even when other apps are fullscreen
  mainWindow.setAlwaysOnTop(true, 'floating', 1);
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  // Global shortcut to toggle visibility (Cmd+Shift+A for Axel)
  globalShortcut.register('CommandOrControl+Shift+A', () => {
    if (isVisible) {
      mainWindow.hide();
      isVisible = false;
    } else {
      mainWindow.show();
      isVisible = true;
    }
  });
}

// Handle minimize from renderer
ipcMain.on('minimize-window', () => {
  mainWindow.hide();
  isVisible = false;
});

ipcMain.on('close-window', () => {
  app.quit();
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  globalShortcut.unregisterAll();
  app.quit();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
