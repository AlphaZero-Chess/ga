import React, { useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { VirtualBrowser } from "./components/VirtualBrowser";
import { Toaster } from "./components/ui/toaster";

const Home = () => {
  const [showBrowser, setShowBrowser] = useState(true);

  return (
    <div className="min-h-screen bg-zinc-950">
      {/* Main content area */}
      <div className="flex flex-col items-center justify-center min-h-screen p-8">
        <div className="text-center max-w-2xl">
          <h1 className="text-4xl font-bold text-zinc-100 mb-4">
            Virtual Chromium Browser
          </h1>
          <p className="text-zinc-400 text-lg mb-8">
            A sophisticated virtual browser with full JavaScript support, 
            extensions management, and search engine integration.
          </p>
          
          {!showBrowser && (
            <button
              onClick={() => setShowBrowser(true)}
              className="px-8 py-3 bg-sky-600 hover:bg-sky-500 text-white font-medium rounded-xl transition-all duration-200 shadow-lg shadow-sky-500/20 hover:shadow-sky-500/30"
            >
              Open Virtual Browser
            </button>
          )}

          <div className="mt-12 grid grid-cols-3 gap-6 text-left">
            <div className="p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800">
              <div className="w-10 h-10 rounded-xl bg-sky-500/20 flex items-center justify-center mb-4">
                <span className="text-sky-400 text-xl">🌐</span>
              </div>
              <h3 className="text-zinc-200 font-semibold mb-2">Full JS Support</h3>
              <p className="text-zinc-500 text-sm">Browse JavaScript-heavy websites with full rendering support</p>
            </div>
            
            <div className="p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800">
              <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center mb-4">
                <span className="text-violet-400 text-xl">🧩</span>
              </div>
              <h3 className="text-zinc-200 font-semibold mb-2">Extensions</h3>
              <p className="text-zinc-500 text-sm">Load unpacked extensions and manage developer tools</p>
            </div>
            
            <div className="p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center mb-4">
                <span className="text-emerald-400 text-xl">🔍</span>
              </div>
              <h3 className="text-zinc-200 font-semibold mb-2">Google Search</h3>
              <p className="text-zinc-500 text-sm">Integrated Google search directly from the address bar</p>
            </div>
          </div>
        </div>
      </div>

      {/* Virtual Browser Panel */}
      {showBrowser && (
        <VirtualBrowser 
          defaultExpanded={true} 
          onClose={() => setShowBrowser(false)} 
        />
      )}
      
      <Toaster />
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
