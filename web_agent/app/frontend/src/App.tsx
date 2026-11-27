import { useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { askQuestion, type AskResponse } from "./api";

const EXAMPLES = ["Where is the Ladin language spoken?", "What is a coral reef made of?"];

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    inFlight.current = new AbortController();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await askQuestion(trimmed, inFlight.current.signal));
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof Error ? e.message : "Something went wrong.");
      }
    } finally {
      setLoading(false);
      inFlight.current = null;
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submit(question);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter submits; Shift+Enter adds a new line.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit(question);
    }
  }

  function askExample(example: string) {
    setQuestion(example);
    void submit(example);
  }

  return (
    <main className="page">
      <header>
        <h1>Wiki Agent</h1>
        <p className="muted">Answers questions from a sample of Wikipedia articles, with the passages it used.</p>
      </header>

      <section aria-labelledby="question-heading">
        <h2 id="question-heading">Question</h2>
        <form onSubmit={onSubmit}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your question…"
            rows={3}
            maxLength={2000}
            disabled={loading}
            aria-label="Question"
          />
          <div className="actions">
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Thinking…" : "Ask"}
            </button>
            {loading && (
              <button type="button" className="secondary" onClick={() => inFlight.current?.abort()}>
                Cancel
              </button>
            )}
            <span className="muted hint">Enter to ask · Shift+Enter for a new line</span>
          </div>
        </form>
        <div className="examples">
          <span className="muted">Try:</span>
          {EXAMPLES.map((example) => (
            <button key={example} type="button" className="chip" onClick={() => askExample(example)} disabled={loading}>
              {example}
            </button>
          ))}
        </div>
      </section>

      <section aria-labelledby="answer-heading" aria-live="polite" aria-busy={loading}>
        <h2 id="answer-heading">Answer</h2>
        {loading && (
          <div className="status">
            <span className="spinner" aria-hidden="true" />
            Searching Wikipedia and thinking… this usually takes a few seconds.
          </div>
        )}
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        {!loading && !error && !result && <p className="muted">The answer will appear here.</p>}
        {result && (
          <>
            <p className="answer">{result.answer}</p>
            <h3>Sources</h3>
            {result.sources.length === 0 ? (
              <p className="muted">No relevant passages were found in the index.</p>
            ) : (
              <ol className="sources">
                {result.sources.map((source) => (
                  <li key={`${source.title}-${source.chunk.slice(0, 40)}`}>
                    <details>
                      <summary>
                        <strong>{source.title}</strong> <span className="muted">score {source.score.toFixed(4)}</span>
                      </summary>
                      <p className="passage">{source.chunk}</p>
                    </details>
                  </li>
                ))}
              </ol>
            )}
          </>
        )}
      </section>
    </main>
  );
}
