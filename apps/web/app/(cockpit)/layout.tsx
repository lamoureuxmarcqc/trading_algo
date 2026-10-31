// app/(cockpit)/layout.tsx â€” layout du cockpit uniquement
import { Sidebar } from "@/components/cockpit/sidebar";

export default function CockpitLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <Sidebar />
      <main className="ml-64">{children}</main>
    </div>
  );
}
