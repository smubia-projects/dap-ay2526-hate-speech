const BASE = (import.meta.env.VITE_API_BASE ?? 'http://localhost:7860').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.status = status
  }
}

function messageFrom(body, fallback) {
  return body?.detail?.message || body?.detail || body?.message || fallback
}

export async function wakeBackend() {
  const response = await fetch(`${BASE}/`)
  if (!response.ok) throw new ApiError('Backend unavailable', response.status)
  return response.json()
}

export async function predict(text, threshold = 0.6) {
  let response
  try {
    response = await fetch(`${BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, threshold }),
    })
  } catch {
    throw new ApiError('Could not reach the demo. The Space may still be waking up.')
  }

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const fallback = response.status === 429
      ? "You're going too fast — slow down and try again."
      : response.status === 503
        ? 'This demo is temporarily paused or warming up. Check back soon.'
        : 'Prediction failed. Please try again.'
    throw new ApiError(messageFrom(body, fallback), response.status)
  }
  return body
}
