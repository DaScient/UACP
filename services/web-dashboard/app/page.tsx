import ClassificationViewer from '@/components/ClassificationViewer';

export default function HomePage() {
  return (
    <main className="container mx-auto px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">
          UAP Intelligence Hub
        </h1>
        <p className="text-gray-400 mt-2">
          Multi-station parallax + multi-modal classification, fully open-source.
        </p>
        <nav className="mt-4 flex flex-wrap gap-3 text-sm" aria-label="Dashboards">
          <a
            href="/UACP/war-gov/"
            className="inline-flex items-center gap-2 rounded border border-gray-700 bg-gray-900/60 px-3 py-2 text-gray-200 hover:border-emerald-400 hover:text-emerald-300"
          >
            🛰️ WAR.GOV/UFO live analytic dashboard
          </a>
          <a
            href="/UACP/widget/"
            className="inline-flex items-center gap-2 rounded border border-gray-700 bg-gray-900/60 px-3 py-2 text-gray-200 hover:border-emerald-400 hover:text-emerald-300"
          >
            🧩 Embeddable UFO·INTEL widget
          </a>
        </nav>
      </header>
      <ClassificationViewer wasmModulePath="/UACP/wasm-engine/uacp_math_engine.js" />
    </main>
  );
}
