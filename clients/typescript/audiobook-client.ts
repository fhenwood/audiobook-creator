/**
 * Audiobook Creator API Client
 * 
 * Auto-generated TypeScript client for the Audiobook Creator REST API.
 * 
 * Usage:
 *   import { AudiobookClient } from './audiobook-client';
 *   const client = new AudiobookClient('http://localhost:7860/api');
 *   
 *   // Create a job
 *   const job = await client.createJob({ title: 'My Book', ... });
 *   
 *   // Stream progress via SSE
 *   client.streamJobProgress(job.job_id, (event) => console.log(event));
 */

// Types
export interface HealthResponse {
    status: string;
    engines: string[];
    version: string;
}

export interface EngineInfo {
    name: string;
    display_name: string;
    voice_count: number;
    capabilities: string[];
    min_vram_gb: number;
    recommended_vram_gb: number;
}

export interface VoiceInfo {
    id: string;
    name: string;
    gender: string;
    description: string;
    tags: string[];
}

export interface JobCreateRequest {
    title: string;
    book_file_path: string;
    engine?: string;
    narrator_voice?: string;
    output_format?: string;
    use_emotion_tags?: boolean;
}

export interface JobStatus {
    job_id: string;
    status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'stalled';
    progress: number;
    message: string;
    created_at: string;
    completed_at?: string;
    output_path?: string;
}

export interface SSEEvent {
    event: 'progress' | 'complete' | 'error' | 'stalled' | 'end';
    job_id: string;
    progress?: number;
    message?: string;
    status?: string;
    output_path?: string;
    timestamp?: string;
}

// Client
export class AudiobookClient {
    private baseUrl: string;

    constructor(baseUrl: string = 'http://localhost:7860/api') {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    // Health
    async health(): Promise<HealthResponse> {
        const res = await fetch(`${this.baseUrl}/health`);
        if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
        return res.json();
    }

    // Engines
    async getEngines(): Promise<EngineInfo[]> {
        const res = await fetch(`${this.baseUrl}/engines`);
        if (!res.ok) throw new Error(`Failed to get engines: ${res.status}`);
        return res.json();
    }

    async getVoices(engineName: string): Promise<VoiceInfo[]> {
        const res = await fetch(`${this.baseUrl}/engines/${engineName}/voices`);
        if (!res.ok) throw new Error(`Failed to get voices: ${res.status}`);
        return res.json();
    }

    // Jobs
    async createJob(request: JobCreateRequest): Promise<JobStatus> {
        const res = await fetch(`${this.baseUrl}/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Failed to create job: ${res.status}`);
        }
        return res.json();
    }

    async getJob(jobId: string): Promise<JobStatus> {
        const res = await fetch(`${this.baseUrl}/jobs/${jobId}`);
        if (!res.ok) throw new Error(`Job not found: ${res.status}`);
        return res.json();
    }

    async listJobs(): Promise<JobStatus[]> {
        const res = await fetch(`${this.baseUrl}/jobs`);
        if (!res.ok) throw new Error(`Failed to list jobs: ${res.status}`);
        return res.json();
    }

    async resumeJob(jobId: string): Promise<JobStatus> {
        const res = await fetch(`${this.baseUrl}/jobs/${jobId}/resume`, {
            method: 'POST',
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Failed to resume job: ${res.status}`);
        }
        return res.json();
    }

    async deleteJob(jobId: string): Promise<void> {
        const res = await fetch(`${this.baseUrl}/jobs/${jobId}`, {
            method: 'DELETE',
        });
        if (!res.ok) throw new Error(`Failed to delete job: ${res.status}`);
    }

    // SSE Progress Streaming
    streamJobProgress(
        jobId: string,
        onEvent: (event: SSEEvent) => void,
        onError?: (error: Error) => void
    ): () => void {
        const eventSource = new EventSource(`${this.baseUrl}/jobs/${jobId}/events`);

        eventSource.onmessage = (e) => {
            try {
                const data: SSEEvent = JSON.parse(e.data);
                onEvent(data);
                if (data.event === 'complete' || data.event === 'error' || data.event === 'end') {
                    eventSource.close();
                }
            } catch (err) {
                onError?.(err as Error);
            }
        };

        eventSource.onerror = () => {
            onError?.(new Error('SSE connection lost'));
            eventSource.close();
        };

        // Return cleanup function
        return () => eventSource.close();
    }

    // Download
    getDownloadUrl(jobId: string): string {
        return `${this.baseUrl}/jobs/${jobId}/download`;
    }
}

// Default export
export default AudiobookClient;
