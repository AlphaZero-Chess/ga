// Mock data for Virtual Browser

export const mockTabs = [
  {
    id: 'tab-1',
    title: 'New Tab',
    url: 'chrome://newtab',
    favicon: null,
    isActive: true,
    isLoading: false
  }
];

export const mockExtensions = [
  {
    id: 'ext-1',
    name: 'React Developer Tools',
    version: '5.0.2',
    description: 'Adds React debugging tools to the Chrome Developer Tools.',
    enabled: true,
    icon: null,
    size: '2.1 MB',
    path: '/extensions/react-devtools'
  },
  {
    id: 'ext-2',
    name: 'Redux DevTools',
    version: '3.1.3',
    description: 'Redux debugging tools for Chrome.',
    enabled: true,
    icon: null,
    size: '1.8 MB',
    path: '/extensions/redux-devtools'
  },
  {
    id: 'ext-3',
    name: 'uBlock Origin',
    version: '1.55.0',
    description: 'An efficient wide-spectrum content blocker.',
    enabled: false,
    icon: null,
    size: '5.2 MB',
    path: '/extensions/ublock'
  }
];

export const mockBookmarks = [
  { id: 'bm-1', title: 'Google', url: 'https://google.com', favicon: 'https://www.google.com/favicon.ico' },
  { id: 'bm-2', title: 'GitHub', url: 'https://github.com', favicon: 'https://github.com/favicon.ico' },
  { id: 'bm-3', title: 'Stack Overflow', url: 'https://stackoverflow.com', favicon: 'https://stackoverflow.com/favicon.ico' }
];

export const mockHistory = [
  { id: 'h-1', title: 'Google Search', url: 'https://google.com', visitedAt: new Date(Date.now() - 3600000) },
  { id: 'h-2', title: 'GitHub - Code', url: 'https://github.com', visitedAt: new Date(Date.now() - 7200000) },
  { id: 'h-3', title: 'Discord', url: 'https://discord.com', visitedAt: new Date(Date.now() - 10800000) }
];

export const mockSearchSuggestions = [
  'discord login',
  'discord download',
  'discord web',
  'discord nitro',
  'discord server'
];

export const defaultNewTabContent = {
  searchEngine: 'Google',
  shortcuts: [
    { name: 'Google', url: 'https://google.com', icon: 'search' },
    { name: 'YouTube', url: 'https://youtube.com', icon: 'play' },
    { name: 'GitHub', url: 'https://github.com', icon: 'github' },
    { name: 'Discord', url: 'https://discord.com', icon: 'message-circle' }
  ]
};

export const browserSettings = {
  developerMode: false,
  darkMode: true,
  searchEngine: 'google',
  homepage: 'chrome://newtab',
  blockPopups: true,
  enableJavaScript: true
};
