const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000/api/rag/universal-search";

export async function universalCardSearch(query) {
  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) throw new Error("API error");
    const data = await res.json();
    // Adapt as needed for your backend's response shape
    return data.results || [];
  } catch (e) {
    return [];
  }
}
