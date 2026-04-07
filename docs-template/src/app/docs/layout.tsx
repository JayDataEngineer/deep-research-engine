"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import Search from "@/components/Search";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-surface">
      <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />
      
      <main className="lg:ml-[280px] p-8 lg:p-12 max-w-5xl mx-auto">
        <header className="flex h-16 items-center justify-between gap-4 mb-20">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 -ml-2 text-on-surface hover:text-primary transition-colors lg:hidden"
            >
              <span className="material-symbols-outlined text-2xl">menu</span>
            </button>
            <div className="hidden lg:block">
               <p className="label-md text-primary opacity-60">TECHNICAL_SERIES_V1</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <Search />
          </div>
        </header>

        <article className="prose-technical max-w-none">
          {children}
        </article>
      </main>
    </div>
  );
}
