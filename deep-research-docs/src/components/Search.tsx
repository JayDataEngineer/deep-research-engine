"use client";

import { useState, useEffect, useRef } from "react";
import Fuse from "fuse.js";
import Link from "next/link";
import { createPortal } from "react-dom";

const dummyDocs = [
  { title: "Introduction", path: "/docs", content: "Overview of RedshiftDB project and its goals.", category: "Core" },
  { title: "Architecture", path: "/docs/architecture", content: "Deep dive into the multi-tenant CRM architecture.", category: "System" },
  { title: "API Reference", path: "/docs/api", content: "Full documentation of all REST API endpoints.", category: "API" },
  { title: "Security Guide", path: "/docs/security", content: "Best practices for securing your deployment.", category: "Security" },
  { title: "Vault KMS", path: "/docs/operations", content: "Configuring HashiCorp Vault for encryption.", category: "Ops" },
  { title: "Workspaces API", path: "/docs/api/workspaces", content: "Managing multi-tenant isolated databases.", category: "API" },
  { title: "Documents API", path: "/docs/api/documents", content: "Core document management features.", category: "API" },
];

export default function Search() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(dummyDocs);
  const inputRef = useRef<HTMLInputElement>(null);

  const fuse = new Fuse(dummyDocs, {
    keys: ["title", "content", "category"],
    threshold: 0.3,
  });

  useEffect(() => {
    if (query.length > 0) {
      setResults(fuse.search(query).map((r) => r.item));
    } else {
      setResults(dummyDocs);
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
        className="w-full lg:max-w-xs flex items-center gap-4 px-4 py-3 bg-surface-container-high border-none transition-all text-xs font-bold font-headline uppercase tracking-widest text-on-surface-variant group hover:bg-surface-container-highest"
      >
        <span className="material-symbols-outlined text-[18px] group-hover:text-primary transition-colors">search</span>
        <span className="flex-1 text-left">Search Documentation...</span>
        <div className="hidden sm:flex items-center gap-1 opacity-40 font-mono text-[9px]">
          CMD+K
        </div>
      </button>

      {isOpen && typeof document !== "undefined" && createPortal(
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] px-4">
          <div 
            className="fixed inset-0 bg-on-surface/40 backdrop-blur-md" 
            onClick={() => setIsOpen(false)}
          />
          
          <div className="w-full max-w-2xl bg-white overflow-hidden relative z-10 animate-in fade-in zoom-in-95 duration-200 border-none shadow-[0_40px_80px_rgba(0,104,117,0.12)]">
            <div className="p-6 border-b border-surface-container flex items-center gap-6">
              <span className="material-symbols-outlined text-primary text-[24px]">search</span>
              <input 
                ref={inputRef}
                autoFocus
                placeholder="TYPE_TO_QUERY_ARCHIVE..."
                className="flex-1 bg-transparent border-none outline-none text-on-surface text-lg font-headline font-bold uppercase tracking-tight placeholder:text-surface-dim"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <button 
                onClick={() => setIsOpen(false)}
                className="text-surface-dim hover:text-on-surface transition-colors"
              >
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>

            <div className="max-h-[60vh] overflow-y-auto p-4 space-y-1">
              {results.length > 0 ? (
                results.map((result) => (
                  <Link 
                    key={result.path} 
                    href={result.path}
                    onClick={() => setIsOpen(false)}
                    className="flex items-center gap-6 p-5 hover:bg-surface-container-low transition-all group"
                  >
                    <div className="w-12 h-12 bg-surface-container flex items-center justify-center text-outline group-hover:bg-primary group-hover:text-on-primary transition-all">
                      <span className="material-symbols-outlined text-[20px]">
                        {result.category === 'API' ? 'code' : 
                         result.category === 'Security' ? 'shield' :
                         result.category === 'Ops' ? 'settings' :
                         'description'}
                      </span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold font-headline uppercase tracking-widest text-xs text-on-surface group-hover:text-primary transition-colors">{result.title}</span>
                        <span className="text-[9px] uppercase tracking-[0.2em] text-outline font-mono">{result.category}</span>
                      </div>
                      <p className="text-xs text-on-surface-variant line-clamp-1 opacity-70">{result.content}</p>
                    </div>
                  </Link>
                ))
              ) : (
                <div className="py-20 text-center text-surface-dim text-[10px] tracking-[0.3em] uppercase font-headline font-bold">
                  No matches found in standard sectors.
                </div>
              )}
            </div>
            
            <div className="p-6 bg-surface-container-low flex items-center justify-between text-[9px] font-bold font-headline text-outline uppercase tracking-widest border-t border-surface-container">
              <div className="flex items-center gap-8">
                <div className="flex items-center gap-3">
                  <span className="px-2 py-1 bg-surface-container-highest text-on-surface">↑↓</span>
                  <span>Navigate</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="px-2 py-1 bg-surface-container-highest text-on-surface">↵</span>
                  <span>Select</span>
                </div>
              </div>
              <span className="opacity-40">RS-SERIES-SEARCH_V2</span>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
