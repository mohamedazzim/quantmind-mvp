/**
 * QuantMind API Proxy Route — App Router catch-all handler.
 *
 * This proxy runs server-side at request time and reads INTERNAL_API_URL
 * from the runtime environment (set in docker-compose as http://api:8000).
 * This avoids the Next.js rewrite limitation where destinations are compiled
 * at build time and cannot use runtime env vars.
 */
import { NextRequest, NextResponse } from "next/server";

// Read at request time — works with Docker runtime env injection
const BACKEND =
  process.env.INTERNAL_API_URL ||
  process.env.NEXT_PUBLIC_API_URL?.replace("/api/v1", "") ||
  "http://127.0.0.1:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params.path, "GET");
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params.path, "POST");
}

export async function PUT(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params.path, "PUT");
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params.path, "DELETE");
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxyRequest(request, params.path, "PATCH");
}

async function proxyRequest(
  request: NextRequest,
  pathSegments: string[],
  method: string
): Promise<NextResponse> {
  const path = pathSegments.join("/");
  const search = request.nextUrl.search || "";
  const targetUrl = `${BACKEND}/api/v1/${path}${search}`;

  // Forward all headers except host
  const headers: Record<string, string> = {};
  request.headers.forEach((value, key) => {
    if (key.toLowerCase() !== "host") {
      headers[key] = value;
    }
  });

  let body: BodyInit | undefined;
  if (method !== "GET" && method !== "DELETE") {
    const contentType = request.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      body = await request.text();
    }
  }

  try {
    const response = await fetch(targetUrl, {
      method,
      headers,
      body,
    });

    const responseHeaders = new Headers();
    response.headers.forEach((value, key) => {
      // Don't forward encoding headers that Next.js handles
      if (!["content-encoding", "transfer-encoding"].includes(key.toLowerCase())) {
        responseHeaders.set(key, value);
      }
    });

    const responseBody = await response.arrayBuffer();
    return new NextResponse(responseBody, {
      status: response.status,
      headers: responseHeaders,
    });
  } catch (err: any) {
    console.error(`[API Proxy] Failed to reach ${targetUrl}:`, err.message);
    return NextResponse.json(
      { detail: "Backend unreachable", message: err.message },
      { status: 502 }
    );
  }
}
