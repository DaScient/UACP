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
      </header>
      <ClassificationViewer wasmModulePath="/UACP/wasm-engine/uacp_math_engine.js" />
    </main>
  );
}
