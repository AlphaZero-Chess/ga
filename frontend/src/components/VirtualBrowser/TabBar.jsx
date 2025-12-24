import React, { useState } from 'react';
import { Plus, ChevronDown } from 'lucide-react';
import BrowserTab from './BrowserTab';
import { cn } from '../../lib/utils';
import { ScrollArea, ScrollBar } from '../ui/scroll-area';

const TabBar = ({ tabs, activeTabId, onTabActivate, onTabClose, onNewTab, onTabsReorder }) => {
  const [draggedTabId, setDraggedTabId] = useState(null);

  const handleDragStart = (e, tabId) => {
    setDraggedTabId(tabId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e, targetTabId) => {
    e.preventDefault();
    if (draggedTabId && draggedTabId !== targetTabId) {
      onTabsReorder(draggedTabId, targetTabId);
    }
    setDraggedTabId(null);
  };

  return (
    <div className="flex items-end bg-zinc-900 border-b border-zinc-700/50 px-2 pt-2">
      {/* Tabs container */}
      <ScrollArea className="flex-1 max-w-full">
        <div className="flex items-end gap-0.5 pb-0">
          {tabs.map((tab) => (
            <BrowserTab
              key={tab.id}
              tab={tab}
              isActive={tab.id === activeTabId}
              onActivate={onTabActivate}
              onClose={onTabClose}
              onDragStart={handleDragStart}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            />
          ))}
        </div>
        <ScrollBar orientation="horizontal" className="h-1" />
      </ScrollArea>
      
      {/* New tab button */}
      <button
        onClick={onNewTab}
        className={cn(
          'flex items-center justify-center w-8 h-8 mb-1 ml-1 rounded-lg',
          'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50',
          'transition-all duration-150'
        )}
        title="New Tab (Ctrl+T)"
      >
        <Plus className="w-4 h-4" />
      </button>

      {/* Tab overflow menu */}
      {tabs.length > 8 && (
        <button
          className={cn(
            'flex items-center justify-center w-8 h-8 mb-1 rounded-lg',
            'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700/50',
            'transition-all duration-150'
          )}
          title="More tabs"
        >
          <ChevronDown className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

export default TabBar;
