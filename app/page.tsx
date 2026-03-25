'use client';

import { useState, useEffect, useCallback } from 'react';

/* ─── Types ─── */
interface QAResponse {
  question: string;
  plan: string | null;
  sub_questions: string[] | null;
  answer: string;
  context: string | null;
}

type ApiStatus = 'checking' | 'online' | 'unhealthy' | 'offline';
type PipeStepState = 'idle' | 'active' | 'done';

const PIPELINE_STEPS = ['plan', 'retrieve', 'summarize', 'verify'] as const;
const STEP_LABELS: Record<string, string> = {
  plan: '🧠 Planning',
  retrieve: '🔍 Retrieval',
  summarize: '✍️ Summarize',
  verify: '✅ Verify',
};

const EXAMPLE_QUESTIONS = [
  'What are vector databases?',
  'How does RAG work?',
  'Explain embeddings vs keywords',
];

/* ─── Component ─── */
export default function Home() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QAResponse | null>(null);
  const [elapsed, setElapsed] = useState('');
  const [error, setError] = useState('');
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [pipeSteps, setPipeSteps] = useState<Record<string, PipeStepState>>(
    Object.fromEntries(PIPELINE_STEPS.map((s) => [s, 'idle']))
  );
  const [showPipeline, setShowPipeline] = useState(false);
  const [showContext, setShowContext] = useState(false);

  // ── Health check on mount ──
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const r = await fetch('/api/health', {
          signal: AbortSignal.timeout(8000),
        });
        setApiStatus(r.ok ? 'online' : 'unhealthy');
      } catch {
        setApiStatus('offline');
      }
    };
    checkHealth();
  }, []);

  // ── Pipeline animation ──
  const animatePipeline = useCallback(() => {
    let stepIdx = 0;

    const reset = Object.fromEntries(
      PIPELINE_STEPS.map((s) => [s, 'idle' as PipeStepState])
    );
    setPipeSteps(reset);
    setShowPipeline(true);

    const timer = setInterval(() => {
      setPipeSteps((prev) => {
        const next = { ...prev };
        // mark previous as done
        if (stepIdx > 0) next[PIPELINE_STEPS[stepIdx - 1]] = 'done';
        // mark current as active
        if (stepIdx < PIPELINE_STEPS.length) {
          next[PIPELINE_STEPS[stepIdx]] = 'active';
        }
        return next;
      });
      stepIdx++;
      if (stepIdx > PIPELINE_STEPS.length) clearInterval(timer);
    }, 2500);

    return () => clearInterval(timer);
  }, []);

  const finishPipeline = () => {
    setPipeSteps(
      Object.fromEntries(
        PIPELINE_STEPS.map((s) => [s, 'done' as PipeStepState])
      )
    );
  };

  // ── Run query ──
  const runQuery = async () => {
    const q = question.trim();
    if (!q) return;

    setLoading(true);
    setResult(null);
    setError('');
    setShowContext(false);

    const cleanup = animatePipeline();
    const t0 = performance.now();

    try {
      const res = await fetch('/api/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API returned ${res.status}: ${txt}`);
      }

      const data: QAResponse = await res.json();
      const secs = ((performance.now() - t0) / 1000).toFixed(1);

      finishPipeline();
      setResult(data);
      setElapsed(secs);
    } catch (err: unknown) {
      finishPipeline();
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      cleanup();
      setLoading(false);
    }
  };

  const clearAll = () => {
    setQuestion('');
    setResult(null);
    setError('');
    setShowPipeline(false);
    setShowContext(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runQuery();
  };

  // ── Derived values ──
  const subQuestions = result?.sub_questions ?? [];
  const searches = subQuestions.length + 1;

  const dotClass =
    apiStatus === 'online'
      ? 'dot dotOk'
      : apiStatus === 'unhealthy'
      ? 'dot dotWarn'
      : apiStatus === 'offline'
      ? 'dot dotErr'
      : 'dot';

  const statusLabel =
    apiStatus === 'checking'
      ? 'Checking API…'
      : apiStatus === 'online'
      ? 'API Online'
      : apiStatus === 'unhealthy'
      ? 'API Unhealthy'
      : 'API Offline';

  return (
    <div className="container">
      {/* ── Header ── */}
      <header className="header">
        <h1>🧠 IKMS Query Planner</h1>
        <p>Multi-Agent RAG with Query Planning &amp; Decomposition</p>
      </header>

      {/* ── Status badges ── */}
      <div className="statusBar">
        <span className="badge">
          <span className={dotClass} /> {statusLabel}
        </span>
        <span className="badge">
          <span className="dot dotOk" /> Gemini LLM
        </span>
        <span className="badge">
          <span className="dot dotOk" /> Pinecone DB
        </span>
      </div>

      {/* ── Query input ── */}
      <div className="card">
        <label className="queryLabel" htmlFor="question">
          ❓ Ask a Question
        </label>
        <textarea
          id="question"
          className="textarea"
          placeholder="e.g. How do vector databases compare to traditional databases and how do they handle scalability?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <div className="examples">
          {EXAMPLE_QUESTIONS.map((eq) => (
            <button
              key={eq}
              className="pill"
              onClick={() => setQuestion(eq)}
            >
              {eq}
            </button>
          ))}
        </div>
        <div className="btnRow">
          <button
            className="btn btnPrimary"
            onClick={runQuery}
            disabled={loading}
          >
            🚀 Run Query
          </button>
          <button className="btn btnGhost" onClick={clearAll}>
            Clear
          </button>
        </div>
      </div>

      {/* ── Pipeline progress ── */}
      {showPipeline && (
        <div className="card">
          <div className="pipeline">
            {PIPELINE_STEPS.map((step, i) => (
              <span key={step}>
                {i > 0 && <span className="pipeArrow">→ </span>}
                <span
                  className={`pipeStep ${
                    pipeSteps[step] === 'active'
                      ? 'pipeStepActive'
                      : pipeSteps[step] === 'done'
                      ? 'pipeStepDone'
                      : ''
                  }`}
                >
                  {STEP_LABELS[step]}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Loader ── */}
      {loading && (
        <div className="loader">
          <div className="spinner" />
          <span className="loaderText">Running multi-agent pipeline…</span>
        </div>
      )}

      {/* ── Error ── */}
      {error && <div className="errorBox">❌ {error}</div>}

      {/* ── Results ── */}
      {result && (
        <>
          {/* Metrics */}
          <div className="metrics">
            <div className="metric">
              <div className="metricVal">{subQuestions.length}</div>
              <div className="metricLabel">Sub-Questions</div>
            </div>
            <div className="metric">
              <div className="metricVal">{searches}</div>
              <div className="metricLabel">Searches</div>
            </div>
            <div className="metric">
              <div className="metricVal">{elapsed}</div>
              <div className="metricLabel">Time (s)</div>
            </div>
          </div>

          {/* Answer */}
          <div className="card">
            <div className="sectionHeader">
              <div className="sectionIcon iconGreen">✅</div>
              <h3>Final Answer</h3>
            </div>
            <div className="answerBody">
              {result.answer || 'No answer returned.'}
            </div>
          </div>

          {/* Plan */}
          {result.plan && (
            <div className="card">
              <div className="sectionHeader">
                <div className="sectionIcon iconBlue">🧠</div>
                <h3>Query Plan</h3>
              </div>
              <div className="planBody">{result.plan}</div>
            </div>
          )}

          {/* Sub-questions */}
          {subQuestions.length > 0 && (
            <div className="card">
              <div className="sectionHeader">
                <div className="sectionIcon iconOrange">🔍</div>
                <h3>Decomposed Sub-Questions</h3>
              </div>
              <div className="subqList">
                {subQuestions.map((sq, i) => (
                  <div className="subqItem" key={i}>
                    <span className="subqNum">{i + 1}</span>
                    <span>{sq}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Context */}
          {result.context && (
            <div className="card">
              <button
                className="contextToggle"
                onClick={() => setShowContext((v) => !v)}
              >
                📚 {showContext ? 'Hide' : 'View'} Retrieved Context
              </button>
              {showContext && (
                <div className="contextBody">{result.context}</div>
              )}
            </div>
          )}
        </>
      )}

      {/* ── Footer ── */}
      <footer className="footer">
        IKMS • Query Planning &amp; Decomposition Agent • Powered by Gemini
        &amp; Pinecone
      </footer>
    </div>
  );
}
