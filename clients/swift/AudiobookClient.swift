/**
 * Audiobook Creator Swift API Client
 * 
 * Swift client for the Audiobook Creator REST API.
 * Compatible with iOS 15+, macOS 12+.
 */

import Foundation

// MARK: - Types

public struct HealthResponse: Codable {
    public let status: String
    public let engines: [String]
    public let version: String
}

public struct EngineInfo: Codable {
    public let name: String
    public let displayName: String
    public let voiceCount: Int
    public let capabilities: [String]
    public let minVramGb: Double
    public let recommendedVramGb: Double
    
    enum CodingKeys: String, CodingKey {
        case name
        case displayName = "display_name"
        case voiceCount = "voice_count"
        case capabilities
        case minVramGb = "min_vram_gb"
        case recommendedVramGb = "recommended_vram_gb"
    }
}

public struct VoiceInfo: Codable {
    public let id: String
    public let name: String
    public let gender: String
    public let description: String
    public let tags: [String]
}

public struct JobCreateRequest: Codable {
    public let title: String
    public let bookFilePath: String
    public var engine: String = "orpheus"
    public var narratorVoice: String = "zac"
    public var outputFormat: String = "m4b"
    public var useEmotionTags: Bool = true
    
    enum CodingKeys: String, CodingKey {
        case title
        case bookFilePath = "book_file_path"
        case engine
        case narratorVoice = "narrator_voice"
        case outputFormat = "output_format"
        case useEmotionTags = "use_emotion_tags"
    }
}

public enum JobStatusType: String, Codable {
    case pending
    case inProgress = "in_progress"
    case completed
    case failed
    case stalled
}

public struct JobStatus: Codable {
    public let jobId: String
    public let status: String
    public let progress: Double
    public let message: String
    public let createdAt: String
    public let completedAt: String?
    public let outputPath: String?
    
    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
        case progress
        case message
        case createdAt = "created_at"
        case completedAt = "completed_at"
        case outputPath = "output_path"
    }
}

public struct SSEEvent: Codable {
    public let event: String
    public let jobId: String
    public let progress: Double?
    public let message: String?
    public let status: String?
    public let outputPath: String?
    public let timestamp: String?
    
    enum CodingKeys: String, CodingKey {
        case event
        case jobId = "job_id"
        case progress
        case message
        case status
        case outputPath = "output_path"
        case timestamp
    }
}

// MARK: - Client

public class AudiobookClient {
    private let baseUrl: URL
    private let session: URLSession
    
    public init(baseUrl: String = "http://localhost:7860/api") {
        self.baseUrl = URL(string: baseUrl)!
        self.session = URLSession.shared
    }
    
    // MARK: - Health
    
    public func health() async throws -> HealthResponse {
        let url = baseUrl.appendingPathComponent("health")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(HealthResponse.self, from: data)
    }
    
    // MARK: - Engines
    
    public func getEngines() async throws -> [EngineInfo] {
        let url = baseUrl.appendingPathComponent("engines")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode([EngineInfo].self, from: data)
    }
    
    public func getVoices(engineName: String) async throws -> [VoiceInfo] {
        let url = baseUrl.appendingPathComponent("engines/\(engineName)/voices")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode([VoiceInfo].self, from: data)
    }
    
    // MARK: - Jobs
    
    public func createJob(request: JobCreateRequest) async throws -> JobStatus {
        var urlRequest = URLRequest(url: baseUrl.appendingPathComponent("jobs"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        
        let (data, _) = try await session.data(for: urlRequest)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }
    
    public func getJob(jobId: String) async throws -> JobStatus {
        let url = baseUrl.appendingPathComponent("jobs/\(jobId)")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }
    
    public func listJobs() async throws -> [JobStatus] {
        let url = baseUrl.appendingPathComponent("jobs")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode([JobStatus].self, from: data)
    }
    
    public func resumeJob(jobId: String) async throws -> JobStatus {
        var urlRequest = URLRequest(url: baseUrl.appendingPathComponent("jobs/\(jobId)/resume"))
        urlRequest.httpMethod = "POST"
        
        let (data, _) = try await session.data(for: urlRequest)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }
    
    public func deleteJob(jobId: String) async throws {
        var urlRequest = URLRequest(url: baseUrl.appendingPathComponent("jobs/\(jobId)"))
        urlRequest.httpMethod = "DELETE"
        let _ = try await session.data(for: urlRequest)
    }
    
    // MARK: - SSE Streaming
    
    public func streamJobProgress(jobId: String) -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            let url = baseUrl.appendingPathComponent("jobs/\(jobId)/events")
            let task = session.dataTask(with: url) { data, response, error in
                if let error = error {
                    continuation.finish(throwing: error)
                    return
                }
                
                guard let data = data,
                      let text = String(data: data, encoding: .utf8) else {
                    continuation.finish()
                    return
                }
                
                // Parse SSE events
                for line in text.components(separatedBy: "\n\n") {
                    if line.hasPrefix("data: ") {
                        let jsonString = String(line.dropFirst(6))
                        if let jsonData = jsonString.data(using: .utf8),
                           let event = try? JSONDecoder().decode(SSEEvent.self, from: jsonData) {
                            continuation.yield(event)
                            if event.event == "complete" || event.event == "error" {
                                continuation.finish()
                                return
                            }
                        }
                    }
                }
            }
            task.resume()
            
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
    
    // MARK: - Download URL
    
    public func getDownloadUrl(jobId: String) -> URL {
        return baseUrl.appendingPathComponent("jobs/\(jobId)/download")
    }
}
