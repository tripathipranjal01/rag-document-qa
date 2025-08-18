'use client';

import { useState, useEffect, useRef } from 'react';
import { Upload, Send, File, MessageSquare, Loader2, CheckCircle, AlertCircle, Eye, Trash2, Clock, FileText, Plus, Search, BookOpen } from 'lucide-react';

interface Document {
  id: string;
  filename: string;
  status: string;
  created_at: string;
  chunk_count?: number;
  error?: string;
}

interface QAResponse {
  answer: string;
  citations: Array<{
    text: string;
    page: number;
    source: string;
    similarity: number;
    chunk_id: string;
  }>;
  sources: Array<{
    page: number;
    source: string;
  }>;
  processing_time: number;
  chunks_used: number;
}

interface DocumentContent {
  id: string;
  filename: string;
  content: string;
  chunks: Array<{
    id: string;
    content: string;
    chunk_index: number;
    page: number;
  }>;
  status: string;
}

export default function Home() {
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState<QAResponse['citations']>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<string>('all');
  const [uploadStatus, setUploadStatus] = useState<{type: 'success' | 'error', message: string} | null>(null);
  const [documentContent, setDocumentContent] = useState<DocumentContent | null>(null);
  const [viewerMode, setViewerMode] = useState<'text' | 'chunks'>('text');
  const [selectedChunk, setSelectedChunk] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'documents'>('chat');
  const [sessionInfo, setSessionInfo] = useState<{session_id: string, document_count: number} | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch documents and session info on component mount
  useEffect(() => {
    fetchSessionInfo();
    fetchDocuments();
  }, []);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://rag-document-qa-fv4c.onrender.com';

  const fetchSessionInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/session`, {
        credentials: 'include',
      });
      if (response.ok) {
        const session = await response.json();
        setSessionInfo(session);
      }
    } catch (error) {
      console.error('Failed to fetch session info:', error);
    }
  };

  const clearSession = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/session`, {
        method: 'DELETE',
        credentials: 'include',
      });
      if (response.ok) {
        setDocuments([]);
        setSessionInfo(null);
        setAnswer('');
        setCitations([]);
        setDocumentContent(null);
        setSelectedDocument('all');
        fetchSessionInfo(); // Get new session
      }
    } catch (error) {
      console.error('Failed to clear session:', error);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include', // Include cookies for session
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Upload result:', result);
        setUploadStatus({ type: 'success', message: `File uploaded successfully: ${result.filename}` });
        
        // Refresh documents list immediately and after a delay
        await fetchDocuments();
        setTimeout(() => fetchDocuments(), 2000);
      } else {
        const error = await response.json();
        console.error('Upload error:', error);
        setUploadStatus({ type: 'error', message: `Upload failed: ${error.detail}` });
      }
    } catch (error) {
      setUploadStatus({ type: 'error', message: 'Upload failed: Network error' });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 5000);
    }
  };

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`, {
        credentials: 'include', // Include cookies for session
      });
      if (response.ok) {
        const docs = await response.json();
        console.log('Fetched documents:', docs);
        setDocuments(docs);
      } else {
        console.error('Failed to fetch documents:', response.status, response.statusText);
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const fetchDocumentContent = async (docId: string) => {
    try {
              const response = await fetch(`${API_BASE_URL}/api/documents/${docId}/content`, {
        credentials: 'include', // Include cookies for session
      });
      if (response.ok) {
        const content = await response.json();
        setDocumentContent(content);
      }
    } catch (error) {
      console.error('Failed to fetch document content:', error);
    }
  };

  const handleDocumentSelect = (docId: string) => {
    setSelectedDocument(docId);
    if (docId !== 'all') {
      fetchDocumentContent(docId);
    } else {
      setDocumentContent(null);
    }
  };

  const handleCitationClick = (chunkId: string) => {
    setSelectedChunk(chunkId);
    setViewerMode('chunks');
  };

  const deleteDocument = async (docId: string) => {
    try {
              const response = await fetch(`${API_BASE_URL}/api/documents/${docId}`, {
        method: 'DELETE',
        credentials: 'include', // Include cookies for session
      });
      if (response.ok) {
        fetchDocuments();
        if (selectedDocument === docId) {
          setSelectedDocument('all');
          setDocumentContent(null);
        }
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  const handleAskQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || documents.length === 0) return;

    setLoading(true);
    setAnswer('');
    setCitations([]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/qa`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // Include cookies for session
        body: JSON.stringify({
          question: question,
          document_id: selectedDocument === 'all' ? null : selectedDocument,
          scope: selectedDocument === 'all' ? 'all' : 'document'
        }),
      });

      if (response.ok) {
        const result: QAResponse = await response.json();
        setAnswer(result.answer);
        setCitations(result.citations);
      } else {
        setAnswer('Failed to get answer. Please try again.');
        setCitations([]);
      }
    } catch (error) {
      setAnswer('Network error. Please check your connection.');
      setCitations([]);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'indexed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'indexed':
        return 'Ready';
      case 'processing':
        return 'Processing';
      case 'failed':
        return 'Failed';
      default:
        return 'Pending';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <BookOpen className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Document Q&A</h1>
                <p className="text-sm text-gray-600">AI-powered document analysis</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {sessionInfo && (
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <span>Session: {sessionInfo.session_id.slice(0, 8)}...</span>
                  <span>•</span>
                  <span>{sessionInfo.document_count} docs</span>
                  <button
                    onClick={clearSession}
                    className="text-red-600 hover:text-red-800 text-xs underline"
                    title="Clear session and all documents"
                  >
                    Clear Session
                  </button>
                </div>
              )}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.bmp,.tiff"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
                ref={fileInputRef}
              />
              <label
                htmlFor="file-upload"
                className="flex items-center px-4 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors cursor-pointer"
              >
                <Plus className="h-4 w-4 mr-2" />
                Upload Document
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Left Sidebar - Documents */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200">
              <div className="px-6 py-4 border-b border-gray-100">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                    <File className="h-5 w-5 mr-2 text-blue-600" />
                    Documents ({documents.length})
                  </h2>
                  <button
                    onClick={fetchDocuments}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    Refresh
                  </button>
                </div>
              </div>
              
              <div className="p-4">
                {documents.length === 0 ? (
                  <div className="text-center py-8">
                    <FileText className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                    <p className="text-gray-500 text-sm">No documents uploaded</p>
                    <p className="text-gray-400 text-xs mt-1">Upload a document to get started</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {/* All Documents Option */}
                    <div
                      className={`flex items-center p-3 rounded-lg cursor-pointer transition-colors ${
                        selectedDocument === 'all' ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => handleDocumentSelect('all')}
                    >
                      <input
                        type="radio"
                        checked={selectedDocument === 'all'}
                        onChange={() => {}}
                        className="mr-3"
                      />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">All Documents</div>
                        <div className="text-xs text-gray-500">Search across all files</div>
                      </div>
                    </div>
                    
                    {/* Individual Documents */}
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        className={`flex items-center p-3 rounded-lg cursor-pointer transition-colors ${
                          selectedDocument === doc.id ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50'
                        }`}
                        onClick={() => handleDocumentSelect(doc.id)}
                      >
                        <input
                          type="radio"
                          checked={selectedDocument === doc.id}
                          onChange={() => {}}
                          className="mr-3"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-gray-900 truncate">{doc.filename}</div>
                          <div className="flex items-center text-xs text-gray-500 mt-1">
                            {getStatusIcon(doc.status)}
                            <span className="ml-1">{getStatusText(doc.status)}</span>
                            <span className="mx-2">•</span>
                            <span>{doc.chunk_count || 0} chunks</span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              fetchDocumentContent(doc.id);
                            }}
                            className="p-1 text-gray-400 hover:text-gray-600"
                            title="View document"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteDocument(doc.id);
                            }}
                            className="p-1 text-gray-400 hover:text-red-600"
                            title="Delete document"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200">
              <div className="px-6 py-4 border-b border-gray-100">
                <div className="flex items-center space-x-6">
                  <button
                    onClick={() => setActiveTab('chat')}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeTab === 'chat' 
                        ? 'bg-blue-100 text-blue-700' 
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    <MessageSquare className="h-5 w-5" />
                    <span>Chat</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('documents')}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                      activeTab === 'documents' 
                        ? 'bg-blue-100 text-blue-700' 
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    <FileText className="h-5 w-5" />
                    <span>Document Viewer</span>
                  </button>
                </div>
              </div>

              <div className="p-6">
                {activeTab === 'chat' ? (
                  /* Chat Interface */
                  <div className="space-y-6">
                    {/* Question Input */}
                    <form onSubmit={handleAskQuestion} className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Ask a question about your documents
                        </label>
                        <div className="flex space-x-3">
                          <textarea
                            placeholder="What would you like to know about your documents?"
                            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none text-gray-900 placeholder-gray-500"
                            rows={3}
                            disabled={loading || documents.length === 0}
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                          />
                          <button
                            type="submit"
                            disabled={loading || documents.length === 0 || !question.trim()}
                            className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
                          >
                            {loading ? (
                              <Loader2 className="h-5 w-5 animate-spin" />
                            ) : (
                              <Send className="h-5 w-5" />
                            )}
                          </button>
                        </div>
                      </div>
                    </form>

                    {/* Answer Display */}
                    {answer && (
                      <div className="bg-gray-50 rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Answer</h3>
                        <div className="prose max-w-none">
                          <p className="text-gray-900 whitespace-pre-wrap leading-relaxed">{answer}</p>
                        </div>
                      </div>
                    )}

                    {/* Citations Display */}
                    {citations.length > 0 && (
                      <div className="bg-blue-50 rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Citations</h3>
                        <div className="space-y-3">
                          {citations.map((citation, index) => (
                            <div
                              key={index}
                              className="bg-white rounded-lg p-4 border border-blue-200 hover:border-blue-300 transition-colors cursor-pointer"
                              onClick={() => {
                                if (citation.chunk_id) {
                                  setSelectedChunk(citation.chunk_id);
                                  setActiveTab('documents');
                                  setViewerMode('chunks');
                                }
                              }}
                            >
                              <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center space-x-2">
                                  <span className="text-sm font-medium text-blue-700">
                                    {citation.source}
                                  </span>
                                  <span className="text-xs text-gray-500">
                                    Page {citation.page}
                                  </span>
                                </div>
                                <span className="text-xs text-gray-500">
                                  {(citation.similarity * 100).toFixed(1)}% match
                                </span>
                              </div>
                              <p className="text-sm text-gray-700 leading-relaxed">
                                {citation.text}
                              </p>
                              <div className="mt-2 text-xs text-blue-600">
                                Click to view in document
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* No Documents Message */}
                    {documents.length === 0 && (
                      <div className="text-center py-12">
                        <Upload className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-gray-900 mb-2">No documents uploaded</h3>
                        <p className="text-gray-500 mb-4">Upload a document or image to start asking questions</p>
                        <p className="text-xs text-gray-400 mb-4">
                          Supports PDF, DOCX, TXT files and images (PNG, JPG) with OCR
                        </p>
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          className="px-4 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
                        >
                          Upload Document
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  /* Document Viewer */
                  <div>
                    {documentContent ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-semibold text-gray-900">
                            {documentContent.filename}
                          </h3>
                          <div className="flex items-center space-x-2">
                            <button
                              onClick={() => setViewerMode('text')}
                              className={`px-3 py-1 rounded text-sm font-medium ${
                                viewerMode === 'text' 
                                  ? 'bg-blue-100 text-blue-700' 
                                  : 'text-gray-600 hover:text-gray-900'
                              }`}
                            >
                              Full Text
                            </button>
                            <button
                              onClick={() => setViewerMode('chunks')}
                              className={`px-3 py-1 rounded text-sm font-medium ${
                                viewerMode === 'chunks' 
                                  ? 'bg-blue-100 text-blue-700' 
                                  : 'text-gray-600 hover:text-gray-900'
                              }`}
                            >
                              Chunks ({documentContent.chunks.length})
                            </button>
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 rounded-lg p-6 max-h-96 overflow-y-auto">
                          {viewerMode === 'text' ? (
                            <div className="prose max-w-none">
                              <pre className="text-sm text-gray-900 whitespace-pre-wrap font-sans">{documentContent.content}</pre>
                            </div>
                          ) : (
                            <div className="space-y-4">
                              {documentContent.chunks.map((chunk, index) => (
                                <div
                                  key={chunk.id}
                                  className={`p-4 rounded-lg border ${
                                    selectedChunk === chunk.id 
                                      ? 'border-blue-300 bg-blue-50' 
                                      : 'border-gray-200 bg-white'
                                  }`}
                                >
                                  <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-gray-700">
                                      Chunk {index + 1}
                                    </span>
                                    <span className="text-xs text-gray-500">
                                      Page {chunk.page}
                                    </span>
                                  </div>
                                  <p className="text-sm text-gray-900">{chunk.content}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12">
                        <FileText className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                        <h3 className="text-lg font-medium text-gray-900 mb-2">No document selected</h3>
                        <p className="text-gray-500">Select a document from the left panel to view its content</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Status */}
      {uploadStatus && (
        <div className="fixed bottom-4 right-4 z-50">
          <div className={`px-4 py-3 rounded-lg shadow-lg ${
            uploadStatus.type === 'success' 
              ? 'bg-green-500 text-white' 
              : 'bg-red-500 text-white'
          }`}>
            {uploadStatus.message}
          </div>
        </div>
      )}
    </div>
  );
}
