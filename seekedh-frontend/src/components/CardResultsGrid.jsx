import React from "react";

export default function CardResultsGrid({ results, loading }) {
  if (loading) {
    return (
      <div className="mt-10 animate-pulse text-center text-gray-400">
        Loading card results...
      </div>
    );
  }
  if (!results.length) {
    return (
      <div className="mt-10 text-center text-gray-500">
        No results yet. Try searching for a card name, mechanic, or combo.
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 w-full max-w-5xl mx-auto mt-6">
      {results.map((card, idx) => (
        <div
          key={card.id || idx}
          className="bg-gray-800 rounded-xl shadow-lg p-4 hover:scale-105 transition transform"
        >
          <div className="mb-2">
            <img
              src={card.image_url || "/default_card.png"}
              alt={card.name}
              className="w-full max-h-96 object-contain rounded-lg"
              loading="lazy"
            />
          </div>
          <div className="font-bold text-xl">{card.name}</div>
          {card.combined_synergy_score !== undefined && (
            <div className="text-xs text-blue-400">
              Synergy: {(card.combined_synergy_score * 100).toFixed(1)}%
            </div>
          )}
          <div className="text-sm text-gray-400">{card.type_line}</div>
          <div className="mt-1 text-gray-300">{card.oracle_text}</div>
          <div className="mt-2 text-xs text-gray-500">Set: {card.set_name || "—"}</div>
        </div>
      ))}
    </div>
  );
}
