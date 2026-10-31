"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Briefcase,
  ShieldCheck,
  Search,
  LineChart,
  Brain,
  Settings,
  LogIn,
} from "lucide-react";

const items = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/portfolio", label: "Portfolio", icon: Briefcase },
  { href: "/risk", label: "Risk", icon: ShieldCheck },
  { href: "/research", label: "Research", icon: Search },
  { href: "/trading", label: "Trading", icon: LineChart },
  { href: "/knowledge", label: "AI Research", icon: Brain },
  { href: "/admin", label: "Admin", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 border-r border-line bg-black/40 backdrop-blur p-4 overflow-y-auto">
      <Link href="/" className="mb-8 block">
        <p className="metric-label">Lamoureux AI</p>
        <h2 className="mt-1 text-lg font-semibold text-white">Platform</h2>
      </Link>
      <nav className="flex flex-col gap-1">
        {items.map((item) => {
          const Icon = item.icon;
          const active =
            pathname === item.href || pathname?.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                active
                  ? "bg-sand text-slate-950 font-semibold"
                  : "text-slate-300 hover:bg-white/5 hover:text-white"
              }`}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="mt-8 border-t border-line pt-4">
        <Link
          href="/login"
          className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-400 hover:bg-white/5 hover:text-white"
        >
          <LogIn className="h-4 w-4" />
          Login
        </Link>
      </div>
    </aside>
  );
}