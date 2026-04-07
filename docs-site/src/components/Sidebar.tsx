"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";

const navGroups = [
  {
    title: "Core Framework",
    items: [
      { name: "Overview", path: "/docs/user-guide", icon: "terminal" },
      { name: "Architecture", path: "/docs/architecture", icon: "hub" },
      { name: "Design Decisions", path: "/docs/architecture/design-decisions", icon: "lightbulb" },
      { name: "Internals", path: "/docs/internals", icon: "memory" },
    ],
  },
  {
    title: "User Guide",
    items: [
      { name: "Getting Started", path: "/docs/user-guide/getting-started", icon: "rocket_launch" },
      { name: "Installation", path: "/docs/user-guide/getting-started/installation", icon: "download" },
      { name: "Quick Start", path: "/docs/user-guide/getting-started/quickstart", icon: "flash_on" },
      { name: "Infrastructure", path: "/docs/user-guide/getting-started/infrastructure", icon: "dns" },
      { name: "Workflows", path: "/docs/user-guide/workflows", icon: "account_tree" },
      { name: "Workers", path: "/docs/user-guide/workers", icon: "engineering" },
      { name: "Storage", path: "/docs/user-guide/storage", icon: "database" },
      { name: "Examples", path: "/docs/user-guide/examples", icon: "description" },
    ],
  },
  {
    title: "Developer Guide",
    items: [
      { name: "Overview", path: "/docs/developer-guide", icon: "code" },
      { name: "AGENTS.md", path: "/docs/developer-guide/agents", icon: "smart_toy" },
      { name: "AI Reference", path: "/docs/developer-guide/ai-reference", icon: "psychology" },
      { name: "Implementation Status", path: "/docs/developer-guide/contributing/implementation-status", icon: "fact_check" },
      { name: "API Reference", path: "/docs/api-reference", icon: "api" },
      { name: "Project Status", path: "/docs/project-status", icon: "query_stats" },
    ],
  },
];

export default function Sidebar({ isOpen, setIsOpen }: { isOpen: boolean; setIsOpen: (val: boolean) => void }) {
  const pathname = usePathname();

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-surface/80 backdrop-blur-md z-40 lg:hidden" 
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside 
        className={`
          fixed inset-y-0 left-0 w-[280px] bg-surface-container/60 backdrop-blur-xl border-r border-outline z-50 
          transition-transform duration-500 cubic-bezier(0.4, 0, 0.2, 1)
          lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'}
          overflow-y-auto no-scrollbar
        `}
      >
        <div className="p-8 flex flex-col min-h-full">
          {/* Brand/System Identifier */}
          <div className="flex items-center gap-4 mb-12 px-2">
            <div className="w-12 h-12 bg-primary/10 border border-primary/20 flex items-center justify-center relative overflow-hidden group">
              <div className="absolute inset-0 bg-primary/5 animate-pulse" />
              <span className="material-symbols-outlined text-primary text-2xl relative z-10">radar</span>
            </div>
            <div>
              <p className="font-headline font-bold text-primary leading-none text-sm uppercase tracking-tighter">DEEP_ENGINE</p>
              <p className="text-[9px] text-matrix font-bold uppercase tracking-[0.2em] mt-1.5 opacity-60 animate-pulse">SYSTEM_ACTIVE_V4</p>
            </div>
          </div>

          <nav className="space-y-10 flex-1">
            {navGroups.map((group) => (
              <div key={group.title}>
                <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant mb-6 px-2 opacity-50 font-headline">
                  {group.title}
                </h3>
                <div className="space-y-1">
                  {group.items.map((item) => {
                    const isActive = pathname === item.path;
                    return (
                      <Link 
                        key={item.name} 
                        href={item.path}
                        onClick={() => setIsOpen(false)}
                        className={`
                          relative group flex items-center gap-4 px-3 py-3 text-[11px] font-bold font-headline uppercase tracking-[0.15em] transition-all duration-300
                          ${isActive 
                            ? 'text-primary' 
                            : 'text-on-surface-variant hover:text-primary'}
                        `}
                      >
                        {isActive && (
                          <motion.div 
                            layoutId="active-nav"
                            className="absolute inset-0 bg-primary/5 border-l-2 border-primary z-0"
                          />
                        )}
                        <span className={`material-symbols-outlined text-[18px] relative z-10 ${isActive ? 'text-primary' : 'group-hover:text-primary transition-colors'}`}>
                          {item.icon}
                        </span>
                        <span className="relative z-10">{item.name}</span>
                      </Link>
                    );
                  })}
                </div>
              </div>
            ))}
          </nav>

          <div className="mt-12 pt-12 border-t border-outline">
            <button className="w-full bg-surface-container-high border border-outline text-primary text-[10px] font-bold py-4 flex items-center justify-center gap-3 hover:bg-primary hover:text-surface transition-all duration-300 uppercase tracking-[0.2em] font-headline group relative overflow-hidden">
               <div className="absolute inset-0 bg-primary opacity-0 group-hover:opacity-10 transition-opacity" />
              <span className="material-symbols-outlined text-sm relative z-10">terminal</span>
              <span className="relative z-10">Access Terminal</span>
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}


