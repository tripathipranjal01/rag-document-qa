'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Send, File, MessageSquare, Loader2, CheckCircle, AlertCircle, Eye, Trash2, Clock, FileText, Plus, Search, BookOpen } from 'lucide-react';

interface Document {
  id: string;
  filename: string;
  status: string;
  created_at: string;
  chunk_count?: number;
  error?: string;
  progress?: number;
  progress_message?: string;
}

interface ProcessingProgress {
  status: string;
  progress: number;
  message: string;
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
  const [uploadingFileName, setUploadingFileName] = useState<string>('');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [processingDocs, setProcessingDocs] = useState<Map<string, ProcessingProgress>>(new Map());
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
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentContentLoading, setDocumentContentLoading] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(true);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

  const fetchSessionInfo = useCallback(async () => {
    setSessionLoading(true);
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
    } finally {
      setSessionLoading(false);
    }
  }, [API_BASE_URL]);

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
        fetchSessionInfo();
      }
    } catch (error) {
      console.error('Failed to clear session:', error);
    }
  };

  // Poll for processing progress
  const pollProgress = async (fileId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/progress/${fileId}`, {
        credentials: 'include',
      });
      if (response.ok) {
        const progress: ProcessingProgress = await response.json();
        setProcessingDocs(prev => new Map(prev.set(fileId, progress)));
        
        // Continue polling if still processing
        if (progress.status !== 'indexed' && progress.status !== 'failed' && progress.progress < 100) {
          setTimeout(() => pollProgress(fileId), 1000);
        } else {
          // Remove from processing map when done - immediately for better UX
          setTimeout(() => {
            setProcessingDocs(prev => {
              const newMap = new Map(prev);
              newMap.delete(fileId);
              return newMap;
            });
          }, 500); // Reduced from 3000ms to 500ms for immediate cleanup
          fetchDocuments();
        }
      }
            } catch (err) {
          console.error('Error polling progress:', err);
        }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadingFileName(file.name);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Upload result:', result);
        setUploadStatus({ type: 'success', message: `File uploaded successfully: ${result.filename}` });
        
        // Start polling for progress
        if (result.file_id) {
          pollProgress(result.file_id);
        }
        
        await fetchDocuments();
      } else {
        const error = await response.json();
        console.error('Upload error:', error);
        let errorMessage = 'Upload failed';
        if (error.detail) {
          if (error.detail.includes('Text extraction failed')) {
            errorMessage = 'Failed to extract text from document. Please check if the file is valid and not corrupted.';
          } else if (error.detail.includes('Text chunking failed')) {
            errorMessage = 'Failed to process document text. The document might be too complex or corrupted.';
          } else if (error.detail.includes('Embedding generation failed')) {
            errorMessage = 'Failed to generate embeddings. Please try again or contact support if the issue persists.';
          } else if (error.detail.includes('File too large')) {
            errorMessage = 'File is too large. Please upload a file smaller than 30MB.';
          } else if (error.detail.includes('File type')) {
            errorMessage = 'Unsupported file type. Please upload PDF, DOCX, TXT, or image files.';
          } else {
            errorMessage = error.detail;
          }
        }
        setUploadStatus({ type: 'error', message: errorMessage });
      }
    } catch (_error) {
      setUploadStatus({ type: 'error', message: 'Upload failed: Network error. Please check your connection and try again.' });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 8000); // Longer display for error messages
    }
  };

  const fetchDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents`, {
        credentials: 'include',
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
    } finally {
      setDocumentsLoading(false);
    }
  }, [API_BASE_URL]);

  // Initialize data on component mount
  useEffect(() => {
    fetchSessionInfo();
    fetchDocuments();
  }, [fetchSessionInfo, fetchDocuments]);

  const fetchDocumentContent = async (docId: string) => {
    setDocumentContentLoading(true);
    try {
      const contentResponse = await fetch(`${API_BASE_URL}/api/documents/${docId}/content`, {
        credentials: 'include',
      });
      
      const chunksResponse = await fetch(`${API_BASE_URL}/api/chunks/${docId}`, {
        credentials: 'include',
      });
      
      if (contentResponse.ok && chunksResponse.ok) {
        const content = await contentResponse.json();
        const chunksData = await chunksResponse.json();
        console.log('Fetched chunks data:', chunksData);
        
        setDocumentContent({
          ...content,
          chunks: chunksData.chunks || []
        });
      } else if (contentResponse.ok) {
        const content = await contentResponse.json();
        console.log('Content response OK but chunks failed:', chunksResponse.status);
        setDocumentContent({
          ...content,
          chunks: []
        });
      } else {
        console.error('Content response failed:', contentResponse.status);
      }
    } catch (error) {
      console.error('Failed to fetch document content:', error);
    } finally {
      setDocumentContentLoading(false);
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

  // Smart scope defaulting - when viewing a document, default Q&A scope to that document
  useEffect(() => {
    if (documentContent && documentContent.id && selectedDocument !== documentContent.id) {
      // Only auto-switch if currently on 'all' - don't override explicit user selection
      if (selectedDocument === 'all') {
        setSelectedDocument(documentContent.id);
      }
    }
  }, [documentContent, selectedDocument]);

  // Citation click handler - used in citation components
  const handleCitationClick = (chunkId: string) => {
    setSelectedChunk(chunkId);
    setViewerMode('chunks');
  };

  const deleteDocument = async (docId: string) => {
    setDeletingDoc(docId);
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/${docId}`, {
        method: 'DELETE',
        credentials: 'include',
      });
      if (response.ok) {
        setQuestion('');
        setAnswer('');
        setCitations([]);
        setDocumentContent(null);
        
        if (selectedDocument === docId) {
          setSelectedDocument('all');
        }
        
        setTimeout(() => {
          fetchDocuments();
        }, 100);
      }
    } catch (error) {
      console.error('Failed to delete document:', error);
    } finally {
      setDeletingDoc(null);
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
        credentials: 'include',
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
    } catch (_error) {
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
              {sessionLoading ? (
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <div className="animate-pulse h-4 bg-gray-200 rounded w-20"></div>
                  <span>•</span>
                  <div className="animate-pulse h-4 bg-gray-200 rounded w-12"></div>
                </div>
              ) : sessionInfo ? (
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <span>Session: {sessionInfo.session_id.slice(0, 8)}...</span>
                  <span>•</span>
                  <span className="flex items-center">
                    {documentsLoading ? (
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    ) : (
                      sessionInfo.document_count
                    )} docs
                  </span>
                  <button
                    onClick={clearSession}
                    disabled={uploading || documentsLoading}
                    className="text-red-600 hover:text-red-800 text-xs underline disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Clear session and all documents"
                  >
                    Clear Session
                  </button>
                </div>
              ) : null}
              <input
                type="file"
                accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.bmp,.tiff"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
                ref={fileInputRef}
                disabled={uploading}
              />
              <label
                htmlFor="file-upload"
                className={`flex items-center px-4 py-2 font-medium rounded-lg transition-all cursor-pointer shadow-sm ${
                  uploading 
                    ? 'bg-gray-400 text-gray-200 cursor-not-allowed' 
                    : 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-md'
                }`}
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    <span className="hidden sm:inline">Uploading</span>
                    <span className="sm:hidden">...</span>
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    <span className="hidden sm:inline">Upload Document</span>
                    <span className="sm:hidden">Upload</span>
                  </>
                )}
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
                    Documents ({documentsLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin mx-1" />
                    ) : (
                      documents.length
                    )})
                  </h2>
                  <button
                    onClick={fetchDocuments}
                    disabled={documentsLoading}
                    className="text-sm text-blue-600 hover:text-blue-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                    title="Refresh document list"
                  >
                    {documentsLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      'Refresh'
                    )}
                  </button>
                </div>
              </div>
              
              <div className="p-4">
                {documentsLoading ? (
                  /* Skeleton Loading for Documents */
                  <div className="space-y-3">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="animate-pulse">
                        <div className="flex items-center p-3 rounded-lg border border-gray-200">
                          <div className="w-4 h-4 bg-gray-200 rounded mr-3"></div>
                          <div className="flex-1">
                            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                            <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                          </div>
                          <div className="flex space-x-1">
                            <div className="w-6 h-6 bg-gray-200 rounded"></div>
                            <div className="w-6 h-6 bg-gray-200 rounded"></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : documents.length === 0 ? (
                  /* Enhanced Empty State */
                  <div className="text-center py-12">
                    <div className="mx-auto w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                      <FileText className="h-12 w-12 text-gray-400" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No documents yet</h3>
                    <p className="text-gray-500 text-sm mb-4">Upload your first document to start asking questions</p>
                    <div className="flex flex-col items-center space-y-2 text-xs text-gray-400">
                      <div className="flex items-center space-x-4">
                        <span className="flex items-center"><FileText className="h-3 w-3 mr-1" />PDF</span>
                        <span className="flex items-center"><FileText className="h-3 w-3 mr-1" />DOCX</span>
                        <span className="flex items-center"><FileText className="h-3 w-3 mr-1" />TXT</span>
                      </div>
                      <span>+ Images with OCR support</span>
                    </div>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Upload Document
                    </button>
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
                    {documents.map((doc) => {
                      const progress = processingDocs.get(doc.id);
                      return (
                        <div key={doc.id}>
                          <div
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
                                {doc.error && (
                                  <>
                                    <span className="mx-2">•</span>
                                    <span className="text-red-500 font-medium">Error</span>
                                  </>
                                )}
                              </div>
                              {doc.error && (
                                <div className="mt-1 text-xs text-red-600 bg-red-50 p-2 rounded border border-red-200">
                                  ⚠️ {doc.error}
                                </div>
                              )}
                            </div>
                            <div className="flex items-center space-x-1">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  fetchDocumentContent(doc.id);
                                }}
                                disabled={documentContentLoading}
                                className="p-1 text-gray-400 hover:text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                                title="View document"
                              >
                                {documentContentLoading ? (
                                  <Loader2 className="h-4 w-4 animate-spin" />
                                ) : (
                                  <Eye className="h-4 w-4" />
                                )}
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  deleteDocument(doc.id);
                                }}
                                disabled={deletingDoc === doc.id}
                                className="p-1 text-gray-400 hover:text-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Delete document"
                              >
                                {deletingDoc === doc.id ? (
                                  <Loader2 className="h-4 w-4 animate-spin text-red-500" />
                                ) : (
                                  <Trash2 className="h-4 w-4" />
                                )}
                              </button>
                            </div>
                          </div>
                          
                          {/* Inline Progress Bar */}
                          {progress && progress.status !== 'indexed' && progress.status !== 'failed' && (
                            <div className="mx-3 mb-2 px-3 py-2 bg-blue-50 rounded-lg border border-blue-200">
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-blue-700 font-medium">{progress.message}</span>
                                <span className="text-blue-600">{progress.progress}%</span>
                              </div>
                              <div className="w-full bg-blue-100 rounded-full h-2">
                                <div 
                                  className="bg-blue-500 h-2 rounded-full transition-all duration-500 ease-out"
                                  style={{ width: `${progress.progress}%` }}
                                ></div>
                              </div>
                              <div className="flex items-center mt-1 text-xs text-blue-600">
                                <div className="w-1 h-1 bg-blue-500 rounded-full animate-pulse mr-1"></div>
                                Processing {progress.status}...
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
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
                          <div className="flex-1">
                            <textarea
                              placeholder={documents.length === 0 ? "Upload a document first to start asking questions..." : "What would you like to know about your documents?"}
                              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none text-gray-900 placeholder-gray-500 disabled:bg-gray-50 disabled:text-gray-500"
                              rows={3}
                              disabled={loading || documents.length === 0 || uploading}
                              value={question}
                              onChange={(e) => setQuestion(e.target.value)}
                            />
                            {documents.length === 0 && (
                              <p className="mt-2 text-xs text-gray-500 flex items-center">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                Upload documents to enable Q&A functionality
                              </p>
                            )}
                          </div>
                          <button
                            type="submit"
                            disabled={loading || documents.length === 0 || !question.trim() || uploading}
                            className="px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center min-w-[80px] justify-center"
                          >
                            {loading ? (
                              <>
                                <Loader2 className="h-5 w-5 animate-spin mr-2" />
                                <span className="hidden sm:inline">Thinking...</span>
                              </>
                            ) : (
                              <>
                                <Send className="h-5 w-5 mr-2" />
                                <span className="hidden sm:inline">Ask</span>
                              </>
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

                    {/* Citations Display - Compact Chips */}
                    {citations.length > 0 && (
                      <div className="bg-blue-50 rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                          Citations
                          <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            {citations.length}
                          </span>
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {citations.map((citation, index) => (
                            <div
                              key={index}
                              className="citation-chip inline-flex items-center bg-white rounded-full px-3 py-2 border border-blue-200 hover:border-blue-400 hover:bg-blue-50 cursor-pointer shadow-sm hover:shadow-md group"
                              onClick={() => {
                                if (citation.chunk_id) {
                                  // Find the document that contains this chunk
                                  // const chunkDoc = documents.find(doc => doc.id === citation.doc_id);
                                  // TODO: Implement proper chunk-to-document mapping
                                  
                                  setSelectedChunk(citation.chunk_id);
                                  setActiveTab('documents');
                                  setViewerMode('chunks');
                                  
                                  // Scroll to the chunk after a brief delay to allow tab switch
                                  setTimeout(() => {
                                    const chunkElement = document.getElementById(`chunk-${citation.chunk_id}`);
                                    if (chunkElement) {
                                      chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                      // Add highlight effect
                                      chunkElement.classList.add('ring-2', 'ring-blue-400', 'ring-opacity-75');
                                      setTimeout(() => {
                                        chunkElement.classList.remove('ring-2', 'ring-blue-400', 'ring-opacity-75');
                                      }, 3000);
                                    }
                                  }, 200);
                                }
                              }}
                              title={`${citation.source} - Page ${citation.page}: ${citation.text.slice(0, 100)}...`}
                            >
                              <div className="flex items-center space-x-2">
                                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                <span className="text-sm font-medium text-blue-700 truncate max-w-[120px]">
                                  {citation.source}
                                </span>
                                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                                  p.{citation.page}
                                </span>
                                <span className="text-xs text-green-600 bg-green-100 px-2 py-0.5 rounded-full">
                                  {(citation.similarity * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        {/* Citation Usage Hint */}
                        <div className="mt-4 flex items-start space-x-2 text-xs text-gray-600 bg-white bg-opacity-50 rounded-lg p-3">
                          <div className="flex-shrink-0 w-4 h-4 rounded-full bg-blue-500 flex items-center justify-center mt-0.5">
                            <span className="text-white text-xs font-bold">!</span>
                          </div>
                          <div>
                            <p className="font-medium text-gray-700 mb-1">Interactive Citations</p>
                            <p>Click any citation chip to jump directly to that section in the document viewer with highlighting.</p>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Loading State for Citations */}
                    {loading && answer && (
                      <div className="bg-blue-50 rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Finding Citations...</h3>
                        <div className="flex space-x-2">
                          {[1, 2, 3].map((i) => (
                            <div key={i} className="animate-pulse">
                              <div className="h-8 bg-blue-200 rounded-full w-24"></div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* No Citations Found */}
                    {answer && !loading && citations.length === 0 && (
                      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
                        <div className="flex items-start space-x-3">
                          <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                          <div>
                            <h3 className="text-sm font-medium text-yellow-800 mb-1">No specific citations found</h3>
                            <p className="text-sm text-yellow-700">
                              The answer was generated from your documents, but no specific text passages met the confidence threshold for citation.
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Enhanced Empty States */}
                    {documents.length === 0 && !loading && (
                      <div className="text-center py-16">
                        <div className="mx-auto w-32 h-32 bg-gradient-to-br from-blue-50 to-indigo-100 rounded-full flex items-center justify-center mb-6">
                          <Upload className="h-16 w-16 text-blue-500" />
                        </div>
                        <h3 className="text-xl font-semibold text-gray-900 mb-3">Ready to analyze your documents</h3>
                        <p className="text-gray-600 mb-6 max-w-md mx-auto leading-relaxed">
                          Upload documents and start asking questions. Our AI will provide detailed answers with precise citations.
                        </p>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-md mx-auto mb-8">
                          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
                            <FileText className="h-6 w-6 text-red-500 mx-auto mb-1" />
                            <span className="text-xs text-gray-600">PDF</span>
                          </div>
                          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
                            <FileText className="h-6 w-6 text-blue-500 mx-auto mb-1" />
                            <span className="text-xs text-gray-600">DOCX</span>
                          </div>
                          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
                            <FileText className="h-6 w-6 text-green-500 mx-auto mb-1" />
                            <span className="text-xs text-gray-600">TXT</span>
                          </div>
                          <div className="bg-white border border-gray-200 rounded-lg p-3 text-center">
                            <Eye className="h-6 w-6 text-purple-500 mx-auto mb-1" />
                            <span className="text-xs text-gray-600">OCR</span>
                          </div>
                        </div>
                        
                        <button
                          onClick={() => fileInputRef.current?.click()}
                          disabled={uploading}
                          className="px-8 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg hover:shadow-xl"
                        >
                          {uploading ? (
                            <>
                              <Loader2 className="h-5 w-5 animate-spin mr-2 inline" />
                              Uploading...
                            </>
                          ) : (
                            <>
                              <Plus className="h-5 w-5 mr-2 inline" />
                              Upload Your First Document
                            </>
                          )}
                        </button>
                      </div>
                    )}
                    
                    {/* No Search Results */}
                    {!answer && !loading && documents.length > 0 && question.trim() && (
                      <div className="text-center py-12">
                        <div className="mx-auto w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                          <Search className="h-10 w-10 text-gray-400" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to search</h3>
                        <p className="text-gray-500 text-sm">Click &quot;Ask&quot; to search through your {documents.length} document{documents.length === 1 ? '' : 's'}</p>
                      </div>
                    )}
                  </div>
                ) : (
                  /* Document Viewer */
                  <div>
                    {documentContentLoading ? (
                      /* Document Loading Skeleton */
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div className="animate-pulse h-6 bg-gray-200 rounded w-64"></div>
                          <div className="flex space-x-2">
                            <div className="animate-pulse h-8 bg-gray-200 rounded w-20"></div>
                            <div className="animate-pulse h-8 bg-gray-200 rounded w-24"></div>
                          </div>
                        </div>
                        <div className="bg-gray-50 rounded-lg p-6 space-y-4">
                          {[1, 2, 3, 4, 5].map((i) => (
                            <div key={i} className="animate-pulse">
                              <div className="h-4 bg-gray-200 rounded w-full mb-2"></div>
                              <div className="h-4 bg-gray-200 rounded w-5/6 mb-2"></div>
                              <div className="h-4 bg-gray-200 rounded w-4/6"></div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : documentContent ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h3 className="text-lg font-semibold text-gray-900">
                            {documentContent.filename}
                          </h3>
                          <div className="flex items-center space-x-2">
                            <button
                              onClick={() => setViewerMode('text')}
                              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                                viewerMode === 'text' 
                                  ? 'bg-blue-100 text-blue-700' 
                                  : 'text-gray-600 hover:text-gray-900'
                              }`}
                            >
                              Full Text
                            </button>
                            <button
                              onClick={() => setViewerMode('chunks')}
                              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                                viewerMode === 'chunks' 
                                  ? 'bg-blue-100 text-blue-700' 
                                  : 'text-gray-600 hover:text-gray-900'
                              }`}
                            >
                              Chunks ({documentContent.chunks?.length || 0})
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
                              {documentContent.chunks?.map((chunk, index) => (
                                <div
                                  key={chunk.id}
                                  id={`chunk-${chunk.id}`}
                                  className={`p-4 rounded-lg border transition-all duration-300 ${
                                    selectedChunk === chunk.id 
                                      ? 'border-blue-300 bg-blue-50 shadow-md' 
                                      : 'border-gray-200 bg-white hover:border-gray-300'
                                  }`}
                                >
                                  <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-gray-700">
                                      Chunk {index + 1}
                                      {selectedChunk === chunk.id && (
                                        <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                          Selected
                                        </span>
                                      )}
                                    </span>
                                    <span className="text-xs text-gray-500">
                                      Page {chunk.page}
                                    </span>
                                  </div>
                                  <p className="text-sm text-gray-900 leading-relaxed">{chunk.content}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      /* Enhanced Document Viewer Empty State */
                      <div className="text-center py-16">
                        <div className="mx-auto w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mb-6">
                          <BookOpen className="h-12 w-12 text-gray-400" />
                        </div>
                        <h3 className="text-xl font-medium text-gray-900 mb-3">Document Viewer</h3>
                        <p className="text-gray-500 mb-6 max-w-sm mx-auto">
                          Select a document from the sidebar to view its content and navigate through text chunks
                        </p>
                        {documents.length > 0 && (
                          <div className="flex flex-wrap justify-center gap-2 max-w-md mx-auto">
                            {documents.slice(0, 3).map((doc) => (
                              <button
                                key={doc.id}
                                onClick={() => {
                                  handleDocumentSelect(doc.id);
                                  setActiveTab('documents');
                                }}
                                className="px-3 py-1.5 bg-blue-50 text-blue-700 text-sm rounded-full hover:bg-blue-100 transition-colors"
                              >
                                {doc.filename.length > 20 ? doc.filename.slice(0, 20) + '...' : doc.filename}
                              </button>
                            ))}
                            {documents.length > 3 && (
                              <span className="px-3 py-1.5 text-gray-500 text-sm">
                                +{documents.length - 3} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Loading Modal */}
      {uploading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-8 shadow-2xl max-w-md w-full mx-4">
            <div className="text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Uploading Document</h3>
              <p className="text-gray-600 text-sm mb-2">
                {uploadingFileName}
              </p>
              <p className="text-gray-600 text-sm">
                Please wait while we process your document. This may take a few moments...
              </p>
              <div className="mt-4 flex items-center justify-center space-x-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Status - Enhanced */}
      {uploadStatus && (
        <div className="fixed bottom-4 right-4 z-50 max-w-md">
          <div className={`px-4 py-3 rounded-lg shadow-lg border-l-4 ${
            uploadStatus.type === 'success' 
              ? 'bg-green-50 text-green-800 border-green-400' 
              : 'bg-red-50 text-red-800 border-red-400'
          }`}>
            <div className="flex items-start">
              <div className="flex-shrink-0">
                {uploadStatus.type === 'success' ? (
                  <CheckCircle className="h-5 w-5 text-green-400" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-red-400" />
                )}
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium">
                  {uploadStatus.type === 'success' ? 'Success' : 'Error'}
                </p>
                <p className="text-sm">
                  {uploadStatus.message}
                </p>
              </div>
              <div className="ml-auto pl-3">
                <button
                  onClick={() => setUploadStatus(null)}
                  className={`inline-flex rounded-md p-1.5 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                    uploadStatus.type === 'success'
                      ? 'text-green-500 hover:bg-green-100 focus:ring-green-600'
                      : 'text-red-500 hover:bg-red-100 focus:ring-red-600'
                  }`}
                >
                  <span className="sr-only">Dismiss</span>
                  <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
