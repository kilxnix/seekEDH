import React, { useState } from "react";
import CardSearchBar from "./components/CardSearchBar";
import CardResultsGrid from "./components/CardResultsGrid";

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center">
      <header className="py-10">
        <h1 className="text-4xl font-extrabold tracking-tight">seekedh</h1>
        <p className="mt-2 text-lg text-gray-400">Universal Card Search</p>
      </header>
      <main className="flex flex-col items-center w-full px-4">
        <CardSearchBar setResults={setResults} setLoading={setLoading} loading={loading} />
        <CardResultsGrid results={results} loading={loading} />
      </main>
      <footer className="mt-auto py-6 text-gray-500 text-sm text-center w-full">
        Built with love for MTG community – seekedh © {new Date().getFullYear()}
      </footer>
    </div>
  );
}

export default App;
