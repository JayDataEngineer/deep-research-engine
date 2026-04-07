"use client";

import { useState } from "react";
import Sidebar from "@/components/Sidebar";
import Search from "@/components/Search";
import { motion } from "framer-motion";

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-surface selection:bg-primary/30 selection:text-primary">
      <Sidebar isOpen={isSidebarOpen} setIsOpen={setIsSidebarOpen} />
      
      <main className="lg:ml-[280px] p-8 lg:p-12 max-w-5xl mx-auto">
        <header className="flex h-16 items-center justify-between gap-4 mb-20 lg:hidden px-4 glass-panel sticky top-0 z-30">
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 -ml-2 text-on-surface hover:text-primary transition-colors"
          >
            <span className="material-symbols-outlined text-2xl">menu</span>
          </button>
          <div className="flex-1">
             <p className="font-headline font-bold text-primary text-xs uppercase tracking-[0.3em]">DEEP_RESEARCH</p>
          </div>
        </header>

        <div className="mb-12 flex justify-end">
          <Search />
        </div>

        <motion.section 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mb-24 mt-12 bg-surface-container/30 border border-outline p-12 relative overflow-hidden blueprint-grid"
        >
          <div className="absolute top-0 right-0 p-4 border-b border-l border-outline bg-surface-container/50 text-[8px] font-bold text-matrix uppercase tracking-[0.4em] z-20">
            SEC_LEVEL_04 // ACCESS_GRANTED
          </div>
          
          <div className="relative z-10">
            <p className="label-md mb-6">SERIES_04 // AUTONOMOUS_INTELLIGENCE</p>
            <h1 className="display-lg mb-8 leading-[0.9]">Deep Research Engine</h1>
            <p className="text-on-surface-variant max-w-2xl leading-relaxed mb-12 text-lg">
              The high-performance orchestration framework for multi-agent intelligence, vector search, and recursive knowledge synthesis.
            </p>
            <div className="flex flex-wrap gap-6">
              <a href="/docs/user-guide" className="relative group overflow-hidden bg-primary text-surface px-10 py-5 font-headline font-bold uppercase tracking-widest text-[11px] transition-all">
                <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-20 transition-opacity" />
                View Documentation
              </a>
              <a href="/docs/architecture" className="border border-outline px-10 py-5 font-headline font-bold uppercase tracking-widest text-[11px] text-primary hover:bg-primary/5 hover:border-primary/50 transition-all">
                System Architecture
              </a>
            </div>
          </div>
        </motion.section>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="p-10 border border-outline bg-surface-container-low/50 backdrop-blur-md relative group overflow-hidden"
          >
            <div className="absolute top-0 left-0 w-1 h-full bg-primary opacity-0 group-hover:opacity-100 transition-opacity" />
            <h3 className="label-md mb-6 text-primary">01 // NEURAL_ORCHESTRATION</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed opacity-80 group-hover:opacity-100 transition-opacity">
              Dynamic graph execution using LangGraph with integrated state management and priority-based worker scheduling.
            </p>
          </motion.div>
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="p-10 border border-outline bg-surface-container-low/50 backdrop-blur-md relative group overflow-hidden"
          >
            <div className="absolute top-0 left-0 w-1 h-full bg-matrix opacity-0 group-hover:opacity-100 transition-opacity" />
            <h3 className="label-md mb-6 text-matrix">02 // KNOWLEDGE_SYNTHESIS</h3>
            <p className="text-sm text-on-surface-variant leading-relaxed opacity-80 group-hover:opacity-100 transition-opacity">
              Multi-modal data ingestion including PDF chunking, ASR/VLM processing, and automated web research with synthesis.
            </p>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

