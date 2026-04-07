import React from "react";
import type { MDXComponents } from "mdx/types";
import { clsx } from "clsx";

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: ({ children }) => (
      <header className="mb-24 mt-12 bg-surface p-0">
        <div className="w-12 h-1 bg-primary-container mb-8"></div>
        <h1 className="text-[3.5rem] font-headline font-bold leading-none tracking-tight text-on-surface">
          {children}
        </h1>
      </header>
    ),
    h2: ({ children }) => (
      <h2 className="text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4">
        <span className="w-2 h-8 bg-primary"></span>
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3">
        <span className="w-6 h-0.5 bg-primary-container"></span>
        {children}
      </h3>
    ),
    p: ({ children }) => (
      <p className="mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl">
        {children}
      </p>
    ),
    code: ({ children, className }) => {
      const isInline = !className || !className.includes('language-');
      return (
        <code 
          className={clsx(
            className,
            "font-mono text-[0.92em] font-bold tracking-tight",
            isInline 
              ? "px-2 py-0.5 bg-on-surface text-[#00e5ff] rounded-none ghost-border mx-0.5" 
              : "text-[#00e5ff] font-medium block leading-relaxed"
          )}
          style={{ color: '#00e5ff' }}
        >
          {children}
        </code>
      );
    },
    pre: ({ children }) => (
      <div className="relative mb-16 group shadow-2xl">
        <div className="absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10">
          TERMINAL_S01 // SRC_LOG
        </div>
        <pre className="p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30">
          <div className="flex items-center gap-2 mb-10 opacity-20">
            <div className="w-1.5 h-1.5 bg-[#00e5ff] rounded-full" />
            <div className="w-1.5 h-1.5 bg-[#00e5ff] rounded-full" />
            <div className="w-1.5 h-1.5 bg-[#00e5ff] rounded-full" />
          </div>
          <div className="font-mono text-[13px] selection:bg-primary/30">
            {children}
          </div>
        </pre>
      </div>
    ),
    ul: ({ children }) => (
      <ul className="mb-10 space-y-4 text-on-surface-variant font-body list-none">
        {children}
      </ul>
    ),
    li: ({ children }) => (
      <li className="flex items-start gap-4">
        <span className="material-symbols-outlined text-primary text-sm mt-1">arrow_forward</span>
        <span>{children}</span>
      </li>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-primary-container bg-surface-container-low p-8 my-16 relative overflow-hidden italic text-on-surface-variant font-body text-lg shadow-sm">
        <div className="absolute top-0 right-0 p-3 text-[9px] text-outline font-bold tracking-widest uppercase opacity-30 font-headline">Editorial_Note</div>
        {children}
      </blockquote>
    ),
    hr: () => (
      <div className="my-24 flex items-center gap-4">
        <hr className="flex-1 border-surface-container" />
        <span className="material-symbols-outlined text-outline-variant text-sm">architecture</span>
        <hr className="flex-1 border-surface-container" />
      </div>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto my-12 border border-surface-container max-w-full no-scrollbar">
        <table className="w-full border-separate border-spacing-0 text-left font-body text-sm min-w-[600px] md:min-w-0">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-surface-container-low text-xs font-headline font-bold tracking-widest uppercase text-on-surface-variant border-b border-surface-container">
        {children}
      </thead>
    ),
    th: ({ children }) => <th className="p-6">{children}</th>,
    td: ({ children }: { children?: React.ReactNode }) => (
      <td className="p-6 text-sm text-on-surface-variant border-b border-surface-container">
        {children}
      </td>
    ),
    Grid: ({ children }: { children?: React.ReactNode }) => (
      <div className="grid md:grid-cols-2 gap-12 my-12 items-start">
        {children}
      </div>
    ),
    Column: ({ children }: { children?: React.ReactNode }) => (
      <div className="flex flex-col h-full">
        {children}
      </div>
    ),
    Divider: () => (
      <div className="my-24 flex items-center gap-4">
        <hr className="flex-1 border-surface-container" />
        <span className="material-symbols-outlined text-outline-variant text-sm">architecture</span>
        <hr className="flex-1 border-surface-container" />
      </div>
    ),
    ...components,
  };
}
