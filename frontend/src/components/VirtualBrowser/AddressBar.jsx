import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  ArrowLeft, ArrowRight, RotateCw, Home, Shield, Star, 
  Search, Lock, AlertTriangle, X, Loader2, Sparkles
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { searchApi } from '../../services/browserApi';

const AddressBar = ({
  url,
  isLoading,
  canGoBack,
  canGoForward,
  isSecure,
  onNavigate,
  onBack,
  onForward,
  onRefresh,
  onStop,
  onHome,
  onBookmark,
  searchSuggestions = [],
  onSearchInput
}) => {
  const [inputValue, setInputValue] = useState(url);
  const [isFocused, setIsFocused] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [localSuggestions, setLocalSuggestions] = useState([]);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef(null);
  const suggestionsTimeoutRef = useRef(null);

  useEffect(() => {
    if (!isFocused) {
      setInputValue(url);
    }
  }, [url, isFocused]);

  // Use either prop suggestions or local suggestions
  const suggestions = searchSuggestions.length > 0 ? searchSuggestions : localSuggestions;

  // Fetch AI suggestions with debounce
  const fetchSuggestions = useCallback(async (query) => {
    if (!query || query.length < 2) {
      setLocalSuggestions([]);
      return;
    }

    // Check if it looks like a URL
    if (query.includes('.') && !query.includes(' ')) {
      setLocalSuggestions([]);
      return;
    }

    if (suggestionsTimeoutRef.current) {
      clearTimeout(suggestionsTimeoutRef.current);
    }

    suggestionsTimeoutRef.current = setTimeout(async () => {
      setIsLoadingSuggestions(true);
      try {
        const result = await searchApi.getSuggestions(query, 6);
        setLocalSuggestions(result.suggestions || []);
      } catch (err) {
        console.error('Failed to get suggestions:', err);
        setLocalSuggestions([]);
      } finally {
        setIsLoadingSuggestions(false);
      }
    }, 300);
  }, []);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setInputValue(value);
    setSelectedIndex(-1);
    
    // Call parent handler if provided
    if (onSearchInput) {
      onSearchInput(value);
    } else {
      // Otherwise fetch locally
      fetchSuggestions(value);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const value = inputValue.trim();
    if (value) {
      // Check if it's a URL or search query
      if (value.includes('.') && !value.includes(' ')) {
        const fullUrl = value.startsWith('http') ? value : `https://${value}`;
        onNavigate(fullUrl);
      } else {
        // Google search
        onNavigate(`https://www.google.com/search?q=${encodeURIComponent(value)}`);
      }
      setShowSuggestions(false);
      setLocalSuggestions([]);
      inputRef.current?.blur();
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInputValue(suggestion);
    onNavigate(`https://www.google.com/search?q=${encodeURIComponent(suggestion)}`);
    setShowSuggestions(false);
    setLocalSuggestions([]);
  };

  const handleKeyDown = (e) => {
    if (suggestions.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < suggestions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev > 0 ? prev - 1 : suggestions.length - 1
      );
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      handleSuggestionClick(suggestions[selectedIndex]);
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
      setSelectedIndex(-1);
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
    setShowSuggestions(true);
    inputRef.current?.select();
  };

  const handleBlur = () => {
    setTimeout(() => {
      setIsFocused(false);
      setShowSuggestions(false);
      setSelectedIndex(-1);
    }, 200);
  };

  const getSecurityIcon = () => {
    if (url?.startsWith('chrome://')) {
      return <Shield className="w-4 h-4 text-zinc-400" />;
    }
    if (isSecure) {
      return <Lock className="w-4 h-4 text-emerald-500" />;
    }
    return <AlertTriangle className="w-4 h-4 text-amber-500" />;
  };

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-zinc-800 border-b border-zinc-700/50">
      {/* Navigation buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={onBack}
          disabled={!canGoBack}
          className={cn(
            'p-2 rounded-full transition-all duration-150',
            canGoBack 
              ? 'text-zinc-300 hover:text-white hover:bg-zinc-700/50' 
              : 'text-zinc-600 cursor-not-allowed'
          )}
          title="Back (Alt+Left)"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        
        <button
          onClick={onForward}
          disabled={!canGoForward}
          className={cn(
            'p-2 rounded-full transition-all duration-150',
            canGoForward 
              ? 'text-zinc-300 hover:text-white hover:bg-zinc-700/50' 
              : 'text-zinc-600 cursor-not-allowed'
          )}
          title="Forward (Alt+Right)"
        >
          <ArrowRight className="w-4 h-4" />
        </button>
        
        <button
          onClick={isLoading ? onStop : onRefresh}
          className="p-2 rounded-full text-zinc-300 hover:text-white hover:bg-zinc-700/50 transition-all duration-150"
          title={isLoading ? 'Stop (Esc)' : 'Refresh (Ctrl+R)'}
        >
          {isLoading ? (
            <X className="w-4 h-4" />
          ) : (
            <RotateCw className="w-4 h-4" />
          )}
        </button>
        
        <button
          onClick={onHome}
          className="p-2 rounded-full text-zinc-300 hover:text-white hover:bg-zinc-700/50 transition-all duration-150"
          title="Home (Alt+Home)"
        >
          <Home className="w-4 h-4" />
        </button>
      </div>

      {/* Address input */}
      <div className="relative flex-1">
        <form onSubmit={handleSubmit} className="relative">
          <div className={cn(
            'flex items-center gap-2 rounded-full px-4 py-2 transition-all duration-200',
            'bg-zinc-900/80 border',
            isFocused ? 'border-sky-500/50 ring-2 ring-sky-500/20' : 'border-zinc-700/50 hover:border-zinc-600'
          )}>
            {/* Security indicator */}
            <div className="flex-shrink-0">
              {getSecurityIcon()}
            </div>
            
            {/* URL input */}
            <input
              ref={inputRef}
              data-address-input
              type="text"
              value={inputValue}
              onChange={handleInputChange}
              onFocus={handleFocus}
              onBlur={handleBlur}
              onKeyDown={handleKeyDown}
              placeholder="Search Google or type a URL"
              className={cn(
                'flex-1 bg-transparent border-none outline-none text-sm',
                'text-zinc-200 placeholder:text-zinc-500'
              )}
            />
            
            {/* Loading indicators */}
            {isLoading && (
              <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
            )}
            {isLoadingSuggestions && !isLoading && (
              <Sparkles className="w-4 h-4 text-violet-400 animate-pulse" />
            )}
          </div>
        </form>

        {/* Search suggestions dropdown */}
        {showSuggestions && suggestions.length > 0 && inputValue && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-50 overflow-hidden">
            <div className="px-3 py-2 border-b border-zinc-700/50 flex items-center gap-2">
              <Sparkles className="w-3 h-3 text-violet-400" />
              <span className="text-xs text-zinc-400">AI-powered suggestions</span>
            </div>
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className={cn(
                  'flex items-center gap-3 w-full px-4 py-2.5 text-left transition-colors',
                  selectedIndex === index 
                    ? 'bg-sky-500/20 text-sky-200' 
                    : 'hover:bg-zinc-700/50 text-zinc-300'
                )}
              >
                <Search className="w-4 h-4 text-zinc-400" />
                <span className="text-sm">{suggestion}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Bookmark button */}
      <button
        onClick={onBookmark}
        className="p-2 rounded-full text-zinc-300 hover:text-amber-400 hover:bg-zinc-700/50 transition-all duration-150"
        title="Bookmark this page (Ctrl+D)"
      >
        <Star className="w-4 h-4" />
      </button>
    </div>
  );
};

export default AddressBar;
