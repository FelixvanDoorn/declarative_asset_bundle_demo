import { afterEach, describe, expect, it, vi } from "vitest";
import { askQuestion } from "./api";

function mockFetch(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("askQuestion", () => {
  it("posts the question and returns the answer with sources", async () => {
    const body = { answer: "Northern Italy.", sources: [{ title: "Ladin language", chunk: "...", score: 0.005 }] };
    const fetchMock = mockFetch(json(body));

    await expect(askQuestion("Where is Ladin spoken?")).resolves.toEqual(body);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/ask",
      expect.objectContaining({ method: "POST", body: JSON.stringify({ question: "Where is Ladin spoken?" }) }),
    );
  });

  it("shows the server's message for handled errors", async () => {
    mockFetch(json({ detail: "The model is busy right now (rate limit). Please try again in a moment." }, 429));
    await expect(askQuestion("hi")).rejects.toThrow("The model is busy right now");
  });

  it("falls back to the status code when the error detail is not a string", async () => {
    mockFetch(json({ detail: [{ msg: "String should have at most 2000 characters" }] }, 422));
    await expect(askQuestion("x")).rejects.toThrow("Request failed (422).");
  });

  it("suggests reloading when the response is not JSON (expired sign-in)", async () => {
    mockFetch(new Response("<html>Sign in</html>", { status: 200, headers: { "Content-Type": "text/html" } }));
    await expect(askQuestion("hi")).rejects.toThrow("Your session may have expired");
  });
});
