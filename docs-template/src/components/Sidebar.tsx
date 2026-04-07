"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navGroups = [
  {
    title: "Documentation",
    items: [
      { name: "Introduction", path: "/docs", icon: "menu_book" },
      { name: "Architecture", path: "/docs/architecture", icon: "architecture" },
      { name: "Security", path: "/docs/security", icon: "shield" },
      { name: "Operations", path: "/docs/operations", icon: "settings" },
    ],
  },
  {
    title: "API Resources",
    items: [
      { name: "API Reference", path: "/docs/api", icon: "api" },
      { name: "Workspaces", path: "/docs/api/workspaces", icon: "layers" },
      { name: "Documents", path: "/docs/api/documents", icon: "description" },
      { name: "Webhooks", path: "/docs/api/webhooks", icon: "webhook" },
    ],
  },
];

export default function Sidebar({ isOpen, setIsOpen }: { isOpen: boolean; setIsOpen: (val: boolean) => void }) {
  const pathname = usePathname();

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-on-surface/20 backdrop-blur-sm z-40 lg:hidden" 
          onClick={() => setIsOpen(false)}
        />
      )}

      <aside 
        className={`
          fixed inset-y-0 left-0 w-[280px] bg-white border-r border-outline-variant/10 z-50 
          transition-transform duration-300 ease-in-out
          lg:translate-x-0 ${isOpen ? 'translate-x-0' : '-translate-x-full'}
          overflow-y-auto no-scrollbar
        `}
      >
        <div className="p-8 border-b-0 flex flex-col min-h-full">
          {/* Profile Section */}
          <div className="flex items-center gap-4 mb-12 px-2">
            <div className="w-12 h-12 bg-surface-container-low flex items-center justify-center overflow-hidden">
              <span className="material-symbols-outlined text-primary text-2xl">person_outline</span>
            </div>
            <div>
              <p className="font-headline font-bold text-on-surface leading-none text-sm uppercase tracking-tighter">The Architect</p>
              <p className="text-[9px] text-outline font-bold uppercase tracking-[0.2em] mt-1.5 opacity-60">TECHNICAL_SERIES_V2</p>
            </div>
          </div>

          <nav className="space-y-10 flex-1">
            {navGroups.map((group) => (
              <div key={group.title}>
                <h3 className="text-[10px] font-bold uppercase tracking-[0.2em] text-outline mb-6 px-2 opacity-50 font-headline">
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
                          flex items-center gap-4 px-3 py-3 text-[11px] font-bold font-headline uppercase tracking-[0.15em] transition-all duration-300
                          ${isActive 
                            ? 'bg-primary/5 text-primary border-l-4 border-primary' 
                            : 'text-on-surface-variant hover:text-primary hover:bg-surface-container-low'}
                        `}
                      >
                        <span className="material-symbols-outlined text-[18px]">
                          {item.icon}
                        </span>
                        {item.name}
                      </Link>
                    );
                  })}
                </div>
              </div>
            ))}
          </nav>

          <div className="mt-12 pt-12 border-t border-outline-variant/10">
            <button className="w-full bg-on-surface text-surface text-[10px] font-bold py-4 flex items-center justify-center gap-3 hover:bg-primary transition-colors uppercase tracking-[0.2em] font-headline">
              <span className="material-symbols-outlined text-sm">picture_as_pdf</span>
              Download Archive
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

