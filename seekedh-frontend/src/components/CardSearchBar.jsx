import React, { useState } from "react";
import { enhancedCardSearch } from "../api";

export default function CardSearchBar({ setResults, setLoading, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    const results = await enhancedCardSearch(query);
    setResults(results);
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-xl flex flex-col gap-3 mb-8">
      <div className="flex gap-2">
        <input
          className="flex-1 p-4 rounded-xl bg-gray-800 border border-gray-700 text-white outline-none text-lg transition-all focus:ring-2 focus:ring-blue-500"
          type="text"
          placeholder="Search for any card, combo, keyword, or synergy..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          autoFocus
        />
        <button
          type="submit"
          className="px-6 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 transition disabled:opacity-50 font-semibold text-lg"
          disabled={loading}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </form>
  );
}
