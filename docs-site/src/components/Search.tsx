"use client";

import { useState, useEffect, useRef } from "react";
import Fuse from "fuse.js";
import Link from "next/link";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";

const projectDocs = [
  { title: "Overview", path: "/docs/user-guide", content: "Deep Research Engine documentation overview.", category: "Core" },
  { title: "Architecture", path: "/docs/architecture", content: "System design and 5-node pipeline architecture.", category: "Core" },
  { title: "Design Decisions", path: "/docs/architecture/design-decisions", content: "Architecture decision records and design docs.", category: "Core" },
  { title: "Internals", path: "/docs/internals", content: "Internal implementation details and integrations.", category: "Core" },
  { title: "Getting Started", path: "/docs/user-guide/getting-started", content: "Setup and installation guide for developers.", category: "User" },
  { title: "Installation", path: "/docs/user-guide/getting-started/installation", content: "Step-by-step installation instructions.", category: "User" },
  { title: "Quick Start", path: "/docs/user-guide/getting-started/quickstart", content: "Run your first workflow in minutes.", category: "User" },
  { title: "Infrastructure", path: "/docs/user-guide/getting-started/infrastructure", content: "Infrastructure requirements and setup.", category: "User" },
  { title: "Workflows", path: "/docs/user-guide/workflows", content: "Configuring and executing multi-agent research flows.", category: "User" },
  { title: "Workers", path: "/docs/user-guide/workers", content: "Implementing and registering custom research workers.", category: "User" },
  { title: "Storage", path: "/docs/user-guide/storage", content: "Configuring Vector and Graph databases.", category: "User" },
  { title: "Examples", path: "/docs/user-guide/examples", content: "Practical examples and usage patterns.", category: "User" },
  { title: "Developer Guide", path: "/docs/developer-guide", content: "Guide for contributors and developers.", category: "Dev" },
  { title: "AGENTS.md", path: "/docs/developer-guide/agents", content: "AI development guide and implementation tasks.", category: "Dev" },
  { title: "AI Reference", path: "/docs/developer-guide/ai-reference", content: "Detailed implementation reference for AIs.", category: "Dev" },
  { title: "API Reference", path: "/docs/api-reference", content: "Complete API documentation.", category: "Dev" },
  { title: "Project Status", path: "/docs/project-status", content: "Current project status and progress.", category: "Dev" },
];

export default function Search() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(projectDocs);
  const inputRef = useRef<HTMLInputElement>(null);

  const fuse = new Fuse(projectDocs, {
    keys: ["title", "content", "category"],
    threshold: 0.3,
  });

  useEffect(() => {
    if (query.length > 0) {
      setResults(fuse.search(query).map((r) => r.item));
    } else {
      setResults(projectDocs);
    }
  }, [query]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  return (
    <>
      <button 
        onClick={() => setIsOpen(true)}
        className="w-full lg:max-w-xs flex items-center gap-4 px-4 py-3 bg-surface-container-high/50 border border-outline/30 backdrop-blur-md transition-all text-xs font-bold font-headline uppercase tracking-widest text-on-surface-variant group hover:bg-surface-container focus:outline-none focus:border-primary/50"
      >
        <span className="material-symbols-outlined text-[18px] group-hover:text-primary transition-colors">search</span>
        <span className="flex-1 text-left group-hover:text-on-surface transition-colors">Search Archive...</span>
        <div className="hidden sm:flex items-center gap-1 opacity-40 font-mono text-[9px] group-hover:opacity-100 transition-opacity">
          CMD+K
        </div>
      </button>

      <AnimatePresence>
        {isOpen && typeof document !== "undefined" && createPortal(
          <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] px-4">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-surface/90 backdrop-blur-2xl" 
              onClick={() => setIsOpen(false)}
            />
            
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: -20 }}
              className="w-full max-w-2xl bg-surface-container-low border border-outline relative z-10 overflow-hidden shadow-[0_0_100px_rgba(0,240,255,0.1)]"
            >
              <div className="p-6 border-b border-outline flex items-center gap-6 relative">
                 <div className="absolute top-0 right-0 py-1 px-4 bg-primary/10 border-b border-l border-outline text-[8px] font-bold text-primary uppercase tracking-[0.3em]">
                   SYSTEM_SEARCH_PROTOCOL
                 </div>
                <span className="material-symbols-outlined text-primary text-[24px]">query_stats</span>
                <input 
                  ref={inputRef}
                  autoFocus
                  placeholder="EXECUTE_QUERY_ON_KNOWLEDGE_BASE..."
                  className="flex-1 bg-transparent border-none outline-none text-primary text-lg font-headline font-bold uppercase tracking-tight placeholder:text-on-surface-variant/30"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button 
                  onClick={() => setIsOpen(false)}
                  className="text-on-surface-variant/50 hover:text-primary transition-colors"
                >
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>

              <div className="max-h-[60vh] overflow-y-auto p-4 space-y-1 no-scrollbar">
                {results.length > 0 ? (
                  results.map((result) => (
                    <Link 
                      key={result.path} 
                      href={result.path}
                      onClick={() => setIsOpen(false)}
                      className="flex items-center gap-6 p-5 hover:bg-primary/5 border border-transparent hover:border-primary/20 transition-all group relative overflow-hidden"
                    >
                      <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-[0.02] transition-opacity" />
                      <div className="w-12 h-12 bg-surface-container/50 border border-outline flex items-center justify-center text-on-surface-variant group-hover:text-primary group-hover:border-primary/40 transition-all relative z-10 shadow-sm">
                        <span className="material-symbols-outlined text-[20px]">
                          {result.category === 'Core' ? 'hub' : 
                           result.category === 'User' ? 'rocket_launch' :
                           result.category === 'Dev' ? 'code' :
                           'description'}
                        </span>
                      </div>
                      <div className="flex-1 relative z-10">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-bold font-headline uppercase tracking-widest text-xs text-on-surface group-hover:text-primary transition-colors">{result.title}</span>
                          <span className="text-[9px] uppercase tracking-[0.2em] text-matrix font-bold font-mono opacity-60">{result.category}</span>
                        </div>
                        <p className="text-xs text-on-surface-variant line-clamp-1 opacity-70 group-hover:opacity-100 transition-opacity leading-relaxed">{result.content}</p>
                      </div>
                    </Link>
                  ))
                ) : (
                  <div className="py-20 text-center text-on-surface-variant/30 text-[10px] tracking-[0.3em] uppercase font-headline font-bold">
                    No matching knowledge sectors identified.
                  </div>
                )}
              </div>
              
              <div className="p-6 bg-surface-container-lowest flex items-center justify-between text-[9px] font-bold font-headline text-on-surface-variant uppercase tracking-widest border-t border-outline">
                <div className="flex items-center gap-8">
                  <div className="flex items-center gap-3">
                    <span className="px-2 py-1 bg-outline/20 text-on-surface border border-outline border-none">↑↓</span>
                    <span>MOVE</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="px-2 py-1 bg-outline/20 text-on-surface border border-outline border-none">↵</span>
                    <span>SELECT</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 bg-matrix animate-pulse rounded-full" />
                  <span className="opacity-40">DEEP_ENGINE_SEARCH_V4</span>
                </div>
              </div>
            </motion.div>
          </div>,
          document.body
        )}
      </AnimatePresence>
    </>
  );
}

