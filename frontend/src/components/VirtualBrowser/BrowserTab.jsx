import React from 'react';
import { X, RotateCw, Globe } from 'lucide-react';
import { cn } from '../../lib/utils';

const BrowserTab = ({ tab, isActive, onActivate, onClose, onDragStart, onDragOver, onDrop }) => {
  return (
    <div
      draggable
      onDragStart={(e) => onDragStart(e, tab.id)}
      onDragOver={onDragOver}
      onDrop={(e) => onDrop(e, tab.id)}
      onClick={() => onActivate(tab.id)}
      className={cn(
        'group flex items-center gap-2 px-3 py-2 min-w-[140px] max-w-[240px] rounded-t-lg cursor-pointer transition-all duration-200',
        'border-x border-t border-transparent hover:border-zinc-600/50',
        isActive 
          ? 'bg-zinc-800 border-zinc-600/50' 
          : 'bg-zinc-900/50 hover:bg-zinc-800/70'
      )}
    >
      {/* Favicon */}
      <div className="flex-shrink-0 w-4 h-4">
        {tab.isLoading ? (
          <RotateCw className="w-4 h-4 text-zinc-400 animate-spin" />
        ) : tab.favicon ? (
          <img src={tab.favicon} alt="" className="w-4 h-4 rounded-sm" />
        ) : (
          <Globe className="w-4 h-4 text-zinc-400" />
        )}
      </div>
      
      {/* Title */}
      <span className="flex-1 text-xs text-zinc-300 truncate font-medium">
        {tab.title || 'New Tab'}
      </span>
      
      {/* Close button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onClose(tab.id);
        }}
        className={cn(
          'flex-shrink-0 p-0.5 rounded-full transition-all duration-150',
          'opacity-0 group-hover:opacity-100',
          'hover:bg-zinc-600/50 text-zinc-400 hover:text-zinc-200'
        )}
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
};

export default BrowserTab;
