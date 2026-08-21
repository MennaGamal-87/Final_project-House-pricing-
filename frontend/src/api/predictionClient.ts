import type { PredictionRequest, PredictionResponse } from "../types/prediction";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {}

export async function predictPrice(payload: PredictionRequest): Promise<PredictionResponse> {
  let response: Response;

  try {
    response = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError("Couldn't reach the prediction service. Check that the backend is running.");
  }

  if (!response.ok) {
    let detail = `Prediction failed (status ${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (Array.isArray(body?.detail)) {
        detail = body.detail.map((d: { msg?: string }) => d.msg).join(" ");
      }
    } catch {
      /* ignore body parse errors, use default detail */
    }
    throw new ApiError(detail);
  }

  return (await response.json()) as PredictionResponse;
}
