"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import Search from "@/components/Search";

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-surface">
      <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />
      
      <main className="lg:ml-[280px] p-8 lg:p-12 max-w-5xl mx-auto">
        <header className="flex h-16 items-center justify-between gap-4 mb-20 lg:hidden">
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 -ml-2 text-on-surface hover:text-primary transition-colors"
          >
            <span className="material-symbols-outlined text-2xl">menu</span>
          </button>
          <div className="flex-1">
             <p className="font-headline font-bold text-on-surface text-sm uppercase tracking-tighter">Documentation</p>
          </div>
        </header>

        <section className="mb-24 mt-12 bg-surface text-on-surface border border-surface-container p-12 relative overflow-hidden blueprint-grid">
          <div className="relative z-10">
            <p className="label-md text-primary mb-4">SERIES_01 // STARTER_TEMPLATE</p>
            <h1 className="display-lg mb-8">Documentation Starter</h1>
            <p className="text-on-surface-variant max-w-2xl leading-relaxed mb-12">
              Welcome to the precision-engineered documentation portal. This template provides the "Editorial Technicality" design system for your projects.
            </p>
            <div className="flex flex-wrap gap-6">
              <a href="/docs" className="bg-on-surface text-surface px-8 py-4 font-headline font-bold uppercase tracking-widest text-[11px] hover:bg-primary transition-all">
                Enter Documentation
              </a>
              <a href="/docs/api" className="border border-outline-variant px-8 py-4 font-headline font-bold uppercase tracking-widest text-[11px] hover:bg-surface-container transition-all">
                API Reference
              </a>
            </div>
          </div>
        </section>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <div className="p-8 border border-surface-container bg-surface-container-lowest">
            <h3 className="label-md mb-6">01 // HIGH PRECISION</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              Standardized on a strict 0px border-radius and high-contrast typography scales for maximum clarity.
            </p>
          </div>
          <div className="p-8 border border-surface-container bg-surface-container-lowest">
            <h3 className="label-md mb-6">02 // MDX POWERED</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              Seamlessly integrate technical specifications, code blocks, and markdown content with zero configuration.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
