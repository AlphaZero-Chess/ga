import React, { useState } from 'react';
import { 
  Package, Upload, FolderArchive, ToggleLeft, ToggleRight, 
  Trash2, Settings, Info, RefreshCw, ChevronRight, Puzzle,
  FolderOpen, FileArchive, CheckCircle2, XCircle
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { Button } from '../ui/button';
import { Switch } from '../ui/switch';
import { ScrollArea } from '../ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from '../ui/dialog';
import { Input } from '../ui/input';
import { Label } from '../ui/label';

const ExtensionsPanel = ({ 
  extensions, 
  developerMode, 
  onToggleDeveloperMode,
  onLoadUnpacked,
  onPackExtension,
  onToggleExtension,
  onRemoveExtension,
  onRefresh
}) => {
  const [loadUnpackedOpen, setLoadUnpackedOpen] = useState(false);
  const [packExtensionOpen, setPackExtensionOpen] = useState(false);
  const [unpackedPath, setUnpackedPath] = useState('');
  const [packPath, setPackPath] = useState('');
  const [packKeyPath, setPackKeyPath] = useState('');

  const handleLoadUnpacked = () => {
    if (unpackedPath.trim()) {
      onLoadUnpacked(unpackedPath.trim());
      setUnpackedPath('');
      setLoadUnpackedOpen(false);
    }
  };

  const handlePackExtension = () => {
    if (packPath.trim()) {
      onPackExtension(packPath.trim(), packKeyPath.trim() || null);
      setPackPath('');
      setPackKeyPath('');
      setPackExtensionOpen(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-zinc-900">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-700/50">
        <div className="flex items-center gap-3">
          <Puzzle className="w-6 h-6 text-sky-400" />
          <h2 className="text-lg font-semibold text-zinc-100">Extensions</h2>
        </div>
        <button
          onClick={onRefresh}
          className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-all"
          title="Refresh extensions"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Developer Mode Toggle */}
      <div className="flex items-center justify-between px-4 py-3 bg-zinc-800/50 border-b border-zinc-700/50">
        <div className="flex items-center gap-3">
          <Settings className="w-4 h-4 text-zinc-400" />
          <span className="text-sm font-medium text-zinc-300">Developer mode</span>
        </div>
        <Switch
          checked={developerMode}
          onCheckedChange={onToggleDeveloperMode}
          className="data-[state=checked]:bg-sky-500"
        />
      </div>

      {/* Developer Actions */}
      {developerMode && (
        <div className="flex gap-2 p-4 border-b border-zinc-700/50 bg-zinc-800/30">
          {/* Load unpacked */}
          <Dialog open={loadUnpackedOpen} onOpenChange={setLoadUnpackedOpen}>
            <DialogTrigger asChild>
              <Button 
                variant="outline" 
                size="sm"
                className="flex items-center gap-2 bg-zinc-800 border-zinc-600 hover:bg-zinc-700 text-zinc-200"
              >
                <FolderOpen className="w-4 h-4" />
                Load unpacked
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-zinc-900 border-zinc-700">
              <DialogHeader>
                <DialogTitle className="text-zinc-100">Load unpacked extension</DialogTitle>
                <DialogDescription className="text-zinc-400">
                  Select the directory containing your extension's manifest.json file.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="extension-path" className="text-zinc-300">Extension directory path</Label>
                  <div className="flex gap-2">
                    <Input
                      id="extension-path"
                      value={unpackedPath}
                      onChange={(e) => setUnpackedPath(e.target.value)}
                      placeholder="/path/to/extension"
                      className="bg-zinc-800 border-zinc-600 text-zinc-200 placeholder:text-zinc-500"
                    />
                    <Button 
                      variant="outline" 
                      size="icon"
                      className="bg-zinc-800 border-zinc-600 hover:bg-zinc-700"
                    >
                      <FolderOpen className="w-4 h-4 text-zinc-300" />
                    </Button>
                  </div>
                </div>
                <div className="text-xs text-zinc-500 flex items-start gap-2">
                  <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>The directory must contain a valid manifest.json file for the extension to load.</span>
                </div>
              </div>
              <DialogFooter>
                <Button 
                  variant="ghost" 
                  onClick={() => setLoadUnpackedOpen(false)}
                  className="text-zinc-400 hover:text-zinc-200"
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handleLoadUnpacked}
                  className="bg-sky-600 hover:bg-sky-500 text-white"
                >
                  Load extension
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Pack extension */}
          <Dialog open={packExtensionOpen} onOpenChange={setPackExtensionOpen}>
            <DialogTrigger asChild>
              <Button 
                variant="outline" 
                size="sm"
                className="flex items-center gap-2 bg-zinc-800 border-zinc-600 hover:bg-zinc-700 text-zinc-200"
              >
                <FileArchive className="w-4 h-4" />
                Pack extension
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-zinc-900 border-zinc-700">
              <DialogHeader>
                <DialogTitle className="text-zinc-100">Pack extension</DialogTitle>
                <DialogDescription className="text-zinc-400">
                  Create a .crx package from an extension directory.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="pack-path" className="text-zinc-300">Extension root directory</Label>
                  <div className="flex gap-2">
                    <Input
                      id="pack-path"
                      value={packPath}
                      onChange={(e) => setPackPath(e.target.value)}
                      placeholder="/path/to/extension"
                      className="bg-zinc-800 border-zinc-600 text-zinc-200 placeholder:text-zinc-500"
                    />
                    <Button 
                      variant="outline" 
                      size="icon"
                      className="bg-zinc-800 border-zinc-600 hover:bg-zinc-700"
                    >
                      <FolderOpen className="w-4 h-4 text-zinc-300" />
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="key-path" className="text-zinc-300">Private key file (optional)</Label>
                  <div className="flex gap-2">
                    <Input
                      id="key-path"
                      value={packKeyPath}
                      onChange={(e) => setPackKeyPath(e.target.value)}
                      placeholder="/path/to/key.pem (leave empty to generate)"
                      className="bg-zinc-800 border-zinc-600 text-zinc-200 placeholder:text-zinc-500"
                    />
                    <Button 
                      variant="outline" 
                      size="icon"
                      className="bg-zinc-800 border-zinc-600 hover:bg-zinc-700"
                    >
                      <FolderOpen className="w-4 h-4 text-zinc-300" />
                    </Button>
                  </div>
                </div>
                <div className="text-xs text-zinc-500 flex items-start gap-2">
                  <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>If no private key is provided, a new one will be generated and saved alongside the .crx file.</span>
                </div>
              </div>
              <DialogFooter>
                <Button 
                  variant="ghost" 
                  onClick={() => setPackExtensionOpen(false)}
                  className="text-zinc-400 hover:text-zinc-200"
                >
                  Cancel
                </Button>
                <Button 
                  onClick={handlePackExtension}
                  className="bg-sky-600 hover:bg-sky-500 text-white"
                >
                  Pack extension
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      )}

      {/* Extensions List */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {extensions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Package className="w-12 h-12 text-zinc-600 mb-4" />
              <p className="text-zinc-400 text-sm">No extensions installed</p>
              <p className="text-zinc-500 text-xs mt-1">Enable Developer mode to load unpacked extensions</p>
            </div>
          ) : (
            extensions.map((ext) => (
              <div
                key={ext.id}
                className={cn(
                  'group p-4 rounded-xl border transition-all duration-200',
                  ext.enabled 
                    ? 'bg-zinc-800/50 border-zinc-700/50 hover:border-zinc-600' 
                    : 'bg-zinc-900/50 border-zinc-800 opacity-60'
                )}
              >
                <div className="flex items-start gap-4">
                  {/* Extension icon */}
                  <div className={cn(
                    'w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0',
                    ext.enabled ? 'bg-sky-500/20' : 'bg-zinc-700/50'
                  )}>
                    <Puzzle className={cn(
                      'w-6 h-6',
                      ext.enabled ? 'text-sky-400' : 'text-zinc-500'
                    )} />
                  </div>
                  
                  {/* Extension info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium text-zinc-200 truncate">{ext.name}</h3>
                      <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
                        v{ext.version}
                      </span>
                    </div>
                    <p className="text-sm text-zinc-400 mt-1 line-clamp-2">{ext.description}</p>
                    
                    {developerMode && (
                      <div className="mt-2 space-y-1">
                        <p className="text-xs text-zinc-500">
                          <span className="font-medium">ID:</span> {ext.id}
                        </p>
                        <p className="text-xs text-zinc-500">
                          <span className="font-medium">Size:</span> {ext.size}
                        </p>
                      </div>
                    )}
                  </div>
                  
                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={ext.enabled}
                      onCheckedChange={() => onToggleExtension(ext.id)}
                      className="data-[state=checked]:bg-sky-500"
                    />
                    <button
                      onClick={() => onRemoveExtension(ext.id)}
                      className="p-2 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-all"
                      title="Remove extension"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
};

export default ExtensionsPanel;
