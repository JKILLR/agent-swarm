'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Folder,
  File,
  FileText,
  FileCode,
  FileImage,
  ChevronRight,
  ChevronDown,
  ArrowLeft,
  Upload,
  Plus,
  Trash2,
  RefreshCw,
  Download,
  Eye,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface FileInfo {
  name: string;
  path: string;
  size: number;
  modified: number;
  is_dir: boolean;
  mime_type: string | null;
  category: string;
  item_count?: number;
}

interface FilesResponse {
  current_path: string;
  workspace: string;
  directories: FileInfo[];
  files: FileInfo[];
}

interface FileContentResponse extends FileInfo {
  content: string;
  encoding: string;
}

interface FileBrowserProps {
  swarmName: string;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileIcon(file: FileInfo) {
  if (file.is_dir) {
    return <Folder className="w-5 h-5 text-yellow-500" />;
  }

  switch (file.category) {
    case 'image':
      return <FileImage className="w-5 h-5 text-purple-400" />;
    case 'text':
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (['py', 'js', 'ts', 'tsx', 'jsx', 'json', 'yaml', 'yml', 'sh'].includes(ext || '')) {
        return <FileCode className="w-5 h-5 text-green-400" />;
      }
      return <FileText className="w-5 h-5 text-blue-400" />;
    default:
      return <File className="w-5 h-5 text-gray-400" />;
  }
}

export default function FileBrowser({ swarmName }: FileBrowserProps) {
  const [currentPath, setCurrentPath] = useState('');
  const [files, setFiles] = useState<FilesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileContentResponse | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showNewFileModal, setShowNewFileModal] = useState(false);

  const fetchFiles = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = currentPath ? `?path=${encodeURIComponent(currentPath)}` : '';
      const response = await fetch(`${API_URL}/api/swarms/${encodeURIComponent(swarmName)}/files${params}`);
      if (!response.ok) {
        throw new Error(`Failed to load files: ${response.statusText}`);
      }
      const data = await response.json();
      setFiles(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files');
    } finally {
      setLoading(false);
    }
  }, [swarmName, currentPath]);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const navigateTo = (path: string) => {
    setCurrentPath(path);
    setSelectedFile(null);
  };

  const navigateUp = () => {
    const parts = currentPath.split('/');
    parts.pop();
    setCurrentPath(parts.join('/'));
    setSelectedFile(null);
  };

  const viewFile = async (file: FileInfo) => {
    try {
      const response = await fetch(
        `${API_URL}/api/swarms/${encodeURIComponent(swarmName)}/files/content?path=${encodeURIComponent(file.path)}`
      );
      if (!response.ok) {
        throw new Error('Failed to load file content');
      }
      const data = await response.json();
      setSelectedFile(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load file');
    }
  };

  const deleteFile = async (file: FileInfo) => {
    if (!confirm(`Delete ${file.name}?`)) return;

    try {
      const response = await fetch(
        `${API_URL}/api/swarms/${encodeURIComponent(swarmName)}/files?path=${encodeURIComponent(file.path)}`,
        { method: 'DELETE' }
      );
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to delete');
      }
      fetchFiles();
      if (selectedFile?.path === file.path) {
        setSelectedFile(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete');
    }
  };

  const breadcrumbs = ['workspace', ...currentPath.split('/').filter(Boolean)];

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Folder className="w-5 h-5 text-yellow-500" />
          <h3 className="font-semibold text-white">Workspace Files</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNewFileModal(true)}
            className="p-2 hover:bg-zinc-800 rounded-lg transition-colors"
            title="New File"
          >
            <Plus className="w-4 h-4 text-zinc-400" />
          </button>
          <button
            onClick={() => setShowUploadModal(true)}
            className="p-2 hover:bg-zinc-800 rounded-lg transition-colors"
            title="Upload File"
          >
            <Upload className="w-4 h-4 text-zinc-400" />
          </button>
          <button
            onClick={fetchFiles}
            className="p-2 hover:bg-zinc-800 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 text-zinc-400 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Breadcrumbs */}
      <div className="flex items-center gap-1 p-3 bg-zinc-950 text-sm overflow-x-auto">
        {currentPath && (
          <button
            onClick={navigateUp}
            className="p-1 hover:bg-zinc-800 rounded transition-colors mr-2"
          >
            <ArrowLeft className="w-4 h-4 text-zinc-400" />
          </button>
        )}
        {breadcrumbs.map((crumb, idx) => (
          <React.Fragment key={idx}>
            {idx > 0 && <ChevronRight className="w-4 h-4 text-zinc-600" />}
            <button
              onClick={() => navigateTo(breadcrumbs.slice(1, idx + 1).join('/'))}
              className={`px-2 py-1 rounded hover:bg-zinc-800 transition-colors ${
                idx === breadcrumbs.length - 1 ? 'text-white' : 'text-zinc-400'
              }`}
            >
              {crumb}
            </button>
          </React.Fragment>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-900/20 border-b border-red-800 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Content */}
      <div className="flex divide-x divide-zinc-800" style={{ height: '400px' }}>
        {/* File List */}
        <div className="flex-1 overflow-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full text-zinc-500">
              Loading...
            </div>
          ) : (
            <div className="divide-y divide-zinc-800/50">
              {files?.directories.map((dir) => (
                <div
                  key={dir.path}
                  className="flex items-center justify-between p-3 hover:bg-zinc-800/50 cursor-pointer group"
                  onClick={() => navigateTo(dir.path)}
                >
                  <div className="flex items-center gap-3">
                    {getFileIcon(dir)}
                    <span className="text-white">{dir.name}</span>
                    <span className="text-xs text-zinc-500">
                      {dir.item_count} items
                    </span>
                  </div>
                  <ChevronRight className="w-4 h-4 text-zinc-600" />
                </div>
              ))}
              {files?.files.map((file) => (
                <div
                  key={file.path}
                  className={`flex items-center justify-between p-3 hover:bg-zinc-800/50 cursor-pointer group ${
                    selectedFile?.path === file.path ? 'bg-zinc-800' : ''
                  }`}
                  onClick={() => viewFile(file)}
                >
                  <div className="flex items-center gap-3">
                    {getFileIcon(file)}
                    <span className="text-white">{file.name}</span>
                    <span className="text-xs text-zinc-500">
                      {formatFileSize(file.size)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        viewFile(file);
                      }}
                      className="p-1 hover:bg-zinc-700 rounded"
                      title="View"
                    >
                      <Eye className="w-4 h-4 text-zinc-400" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteFile(file);
                      }}
                      className="p-1 hover:bg-red-900/50 rounded"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4 text-red-400" />
                    </button>
                  </div>
                </div>
              ))}
              {files?.directories.length === 0 && files?.files.length === 0 && (
                <div className="flex flex-col items-center justify-center h-64 text-zinc-500">
                  <Folder className="w-12 h-12 mb-2 opacity-50" />
                  <p>Empty directory</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* File Preview */}
        {selectedFile && (
          <div className="w-1/2 flex flex-col bg-zinc-950">
            <div className="flex items-center justify-between p-3 border-b border-zinc-800">
              <span className="text-sm text-white font-medium truncate">
                {selectedFile.name}
              </span>
              <button
                onClick={() => setSelectedFile(null)}
                className="text-zinc-500 hover:text-white"
              >
                ×
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {selectedFile.category === 'image' ? (
                <img
                  src={`data:${selectedFile.mime_type};base64,${selectedFile.content}`}
                  alt={selectedFile.name}
                  className="max-w-full h-auto"
                />
              ) : selectedFile.category === 'text' ? (
                <pre className="text-sm text-zinc-300 whitespace-pre-wrap font-mono">
                  {selectedFile.encoding === 'utf-8'
                    ? selectedFile.content
                    : atob(selectedFile.content)}
                </pre>
              ) : (
                <div className="text-zinc-500 text-center py-8">
                  Binary file - {formatFileSize(selectedFile.size)}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <UploadModal
          swarmName={swarmName}
          currentPath={currentPath}
          onClose={() => setShowUploadModal(false)}
          onSuccess={() => {
            setShowUploadModal(false);
            fetchFiles();
          }}
        />
      )}

      {/* New File Modal */}
      {showNewFileModal && (
        <NewFileModal
          swarmName={swarmName}
          currentPath={currentPath}
          onClose={() => setShowNewFileModal(false)}
          onSuccess={() => {
            setShowNewFileModal(false);
            fetchFiles();
          }}
        />
      )}
    </div>
  );
}

interface ModalProps {
  swarmName: string;
  currentPath: string;
  onClose: () => void;
  onSuccess: () => void;
}

function UploadModal({ swarmName, currentPath, onClose, onSuccess }: ModalProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    setError(null);

    try {
      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('path', currentPath);

        const response = await fetch(
          `${API_URL}/api/swarms/${encodeURIComponent(swarmName)}/files`,
          {
            method: 'POST',
            body: formData,
          }
        );

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || 'Upload failed');
        }
      }
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg w-full max-w-md p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Upload Files</h3>

        {error && (
          <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded text-red-400 text-sm">
            {error}
          </div>
        )}

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700'
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleUpload(e.dataTransfer.files);
          }}
        >
          <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-2" />
          <p className="text-zinc-400 mb-2">
            Drag files here or click to browse
          </p>
          <input
            type="file"
            multiple
            className="hidden"
            id="file-upload"
            onChange={(e) => handleUpload(e.target.files)}
          />
          <label
            htmlFor="file-upload"
            className="inline-block px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded cursor-pointer transition-colors"
          >
            Browse Files
          </label>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            disabled={uploading}
            className="px-4 py-2 text-zinc-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
        </div>

        {uploading && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-lg">
            <RefreshCw className="w-8 h-8 text-white animate-spin" />
          </div>
        )}
      </div>
    </div>
  );
}

function NewFileModal({ swarmName, currentPath, onClose, onSuccess }: ModalProps) {
  const [filename, setFilename] = useState('');
  const [content, setContent] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!filename.trim()) {
      setError('Filename is required');
      return;
    }

    setCreating(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('filename', filename);
      formData.append('content', content);
      formData.append('path', currentPath);
      formData.append('is_text', 'true');

      const response = await fetch(
        `${API_URL}/api/swarms/${encodeURIComponent(swarmName)}/files`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create file');
      }

      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create file');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg w-full max-w-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Create New File</h3>

        {error && (
          <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded text-red-400 text-sm">
            {error}
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1">Filename</label>
            <input
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              placeholder="example.txt"
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded focus:outline-none focus:border-blue-500 text-white"
            />
          </div>

          <div>
            <label className="block text-sm text-zinc-400 mb-1">Content (optional)</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={8}
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded focus:outline-none focus:border-blue-500 text-white font-mono text-sm resize-none"
              placeholder="File content..."
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            disabled={creating}
            className="px-4 py-2 text-zinc-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={creating}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors disabled:opacity-50"
          >
            {creating ? 'Creating...' : 'Create File'}
          </button>
        </div>
      </div>
    </div>
  );
}
