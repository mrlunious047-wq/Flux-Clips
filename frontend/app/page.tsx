'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { 
  Upload, 
  Video, 
  Scissors, 
  Sparkles, 
  Download, 
  Play,
  Clock,
  CheckCircle,
  Loader2,
  Share2,
  Settings,
  Zap
} from 'lucide-react';
import ClipGallery from '@/components/ClipGallery';
import PromptInput from '@/components/PromptInput';
import ProcessingStatus from '@/components/ProcessingStatus';
import VideoPlayer from '@/components/VideoPlayer';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Clip {
  id: string;
  start_time: number;
  end_time: number;
  duration: number;
  caption: string;
  score: number;
  video_url?: string;
  thumbnail_url?: string;
}

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobId, setJobId] = useState<string>('');
  const [clips, setClips] = useState<Clip[]>([]);
  const [prompt, setPrompt] = useState('Find the most engaging moments that would go viral on social media');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;
    
    setVideoFile(file);
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post(`${API_BASE}/api/upload`, formData, {
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
          setUploadProgress(percent);
        }
      });
      
      if (response.data.success) {
        setVideoUrl(response.data.url);
        setIsUploading(false);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      setIsUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    },
    maxFiles: 1,
    maxSize: 2 * 1024 * 1024 * 1024 // 2GB
  });

  const handleProcess = async () => {
    if (!videoUrl) {
      alert('Please upload a video first');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const response = await axios.post(`${API_BASE}/api/process`, {
        video_url: videoUrl,
        prompt: prompt,
        num_clips: 5,
        min_duration: 15,
        max_duration: 60,
        style: 'engaging'
      });
      
      if (response.data.success) {
        setJobId(response.data.job_id);
        pollJobStatus(response.data.job_id);
      }
    } catch (error) {
      console.error('Processing failed:', error);
      setIsProcessing(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/jobs/${jobId}`);
        const job = response.data;
        
        if (job.status === 'completed') {
          clearInterval(interval);
          setIsProcessing(false);
          setClips(job.clips || []);
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsProcessing(false);
          alert('Processing failed: ' + job.error);
        }
      } catch (error) {
        console.error('Polling error:', error);
        clearInterval(interval);
        setIsProcessing(false);
      }
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg">
                <Scissors className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">ClipMaker Pro</h1>
                <p className="text-sm text-gray-400">AI-powered video clipping</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="flex items-center space-x-2 px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition">
                <Settings className="h-4 w-4" />
                <span>Settings</span>
              </button>
              <button className="px-6 py-2 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold hover:opacity-90 transition">
                Upgrade
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Upload & Controls */}
          <div className="lg:col-span-2 space-y-8">
            {/* Upload Section */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
              <div className="flex items-center mb-6">
                <div className="p-2 bg-blue-500/20 rounded-lg mr-3">
                  <Upload className="h-5 w-5 text-blue-400" />
                </div>
                <h2 className="text-xl font-bold">Upload Video</h2>
              </div>
              
              {!videoUrl ? (
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                    isDragActive 
                      ? 'border-blue-500 bg-blue-500/10' 
                      : 'border-gray-600 hover:border-blue-500 hover:bg-gray-800/50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <div className="mb-4">
                    <div className="inline-flex p-4 bg-gray-900 rounded-full">
                      <Video className="h-8 w-8 text-gray-400" />
                    </div>
                  </div>
                  <p className="text-lg font-medium mb-2">
                    {isDragActive ? 'Drop video here' : 'Drag & drop video file'}
                  </p>
                  <p className="text-gray-400 mb-4">
                    MP4, MOV, AVI up to 2GB
                  </p>
                  <button className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-semibold hover:opacity-90 transition">
                    Select File
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-900 rounded-lg">
                    <div className="flex items-center space-x-4">
                      <Video className="h-8 w-8 text-green-500" />
                      <div>
                        <p className="font-medium">Video uploaded successfully</p>
                        <p className="text-sm text-gray-400">
                          {videoFile?.name} • {videoFile && (videoFile.size / (1024 * 1024)).toFixed(1)}MB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setVideoFile(null);
                        setVideoUrl('');
                        setUploadProgress(0);
                      }}
                      className="text-red-400 hover:text-red-300"
                    >
                      Remove
                    </button>
                  </div>
                  
                  <VideoPlayer src={videoUrl} />
                </div>
              )}
              
              {isUploading && (
                <div className="mt-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-sm">Uploading...</span>
                    <span className="text-sm font-medium">{uploadProgress}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* AI Prompt Section */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
              <div className="flex items-center mb-6">
                <div className="p-2 bg-purple-500/20 rounded-lg mr-3">
                  <Sparkles className="h-5 w-5 text-purple-400" />
                </div>
                <h2 className="text-xl font-bold">AI Instructions</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Describe what clips you want
                  </label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full h-32 bg-gray-900 border border-gray-700 rounded-lg p-4 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Example: Find funny moments, epic fails, inspiring quotes, or educational insights..."
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Number of clips
                    </label>
                    <select className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500">
                      <option>5 clips</option>
                      <option>10 clips</option>
                      <option>15 clips</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Style
                    </label>
                    <select className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-blue-500">
                      <option>Engaging</option>
                      <option>Educational</option>
                      <option>Funny</option>
                      <option>Dramatic</option>
                    </select>
                  </div>
                </div>
                
                <button
                  onClick={handleProcess}
                  disabled={!videoUrl || isProcessing}
                  className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl font-bold text-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-3"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="h-5 w-5" />
                      <span>Generate Clips with AI</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Processing Status */}
            {jobId && (
              <ProcessingStatus jobId={jobId} />
            )}

            {/* Generated Clips */}
            {clips.length > 0 && (
              <ClipGallery clips={clips} />
            )}
          </div>

          {/* Right Panel - Preview & Stats */}
          <div className="space-y-8">
            {/* Stats Card */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
              <h3 className="text-lg font-bold mb-4">Statistics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Total Videos Processed</span>
                  <span className="font-bold">1,247</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Clips Generated</span>
                  <span className="font-bold">6,235</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Time Saved</span>
                  <span className="font-bold">420 hours</span>
                </div>
                <div className="pt-4 border-t border-gray-700">
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-green-500 to-blue-500 w-3/4" />
                  </div>
                  <p className="text-sm text-gray-400 mt-2">AI Accuracy: 87%</p>
                </div>
              </div>
            </div>

            {/* Quick Tips */}
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl border border-gray-700 p-6">
              <h3 className="text-lg font-bold mb-4 flex items-center">
                <Sparkles className="h-5 w-5 mr-2 text-yellow-400" />
                Pro Tips
              </h3>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">Use specific prompts like "funny moments" or "key insights"</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">Keep clips between 15-60 seconds for social media</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                  <span className="text-sm">Upload HD videos (720p+) for best results</span>
                </li>
              </ul>
            </div>

            {/* Recent Activity */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl border border-gray-700 p-6">
              <h3 className="text-lg font-bold mb-4">Recent Activity</h3>
              <div className="space-y-4">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="flex items-center space-x-3 p-3 bg-gray-900/50 rounded-lg">
                    <div className="p-2 bg-blue-500/20 rounded-lg">
                      <Scissors className="h-4 w-4 text-blue-400" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium">Video processed</p>
                      <p className="text-xs text-gray-400">5 clips generated • 2 min ago</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-12">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="text-gray-400 text-sm">
              © 2024 ClipMaker Pro. All rights reserved.
            </div>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <a href="#" className="text-gray-400 hover:text-white transition">Privacy</a>
              <a href="#" className="text-gray-400 hover:text-white transition">Terms</a>
              <a href="#" className="text-gray-400 hover:text-white transition">API</a>
              <a href="#" className="text-gray-400 hover:text-white transition">Support</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
