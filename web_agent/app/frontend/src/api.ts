export type Source = { title: string; chunk: string; score: number };
export type AskResponse = { answer: string; sources: Source[] };

/** POST the question to the FastAPI backend; throws an Error with a user-facing message. */
export async function askQuestion(question: string, signal?: AbortSignal): Promise<AskResponse> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok || body === null) {
    if (body === null) {
      // Not JSON: usually the Databricks sign-in page after the session expired.
      throw new Error("Unexpected response from the server. Your session may have expired — reload the page.");
    }
    const detail = (body as { detail?: unknown }).detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status}).`);
  }
  return body as AskResponse;
}
