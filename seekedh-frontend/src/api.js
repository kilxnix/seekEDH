const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const UNIVERSAL_URL = `${BASE_URL}/api/rag/universal-search`;
const ENHANCED_URL = `${BASE_URL}/api/rag/enhanced-search`;

/**
 * Query the universal-search endpoint.
 *
 * The backend returns objects under the `cards` key and supports an
 * `include_images` flag to embed image information.  The testing suite
 * uses this flag so we mirror that behaviour here.
 */
export async function universalCardSearch(query) {
  try {
    const res = await fetch(UNIVERSAL_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, include_images: true }),
    });
    if (!res.ok) throw new Error("API error");
    const data = await res.json();
    return data.cards || [];
  } catch (e) {
    console.error("search error", e);
    return [];
  }
}

/**
 * Query the enhanced-search endpoint with optional filters.
 */
export async function enhancedCardSearch(query, filters = {}) {
  try {
    const res = await fetch(ENHANCED_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, filters, include_images: true }),
    });
    if (!res.ok) throw new Error("API error");
    const data = await res.json();
    return data.cards || [];
  } catch (e) {
    console.error("enhanced search error", e);
    return [];
  }
}
