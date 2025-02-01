import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { io, Socket } from 'socket.io-client';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement', { static: true }) videoElementRef!: ElementRef<HTMLVideoElement>;
  private canvas: HTMLCanvasElement;
  private socket!: Socket;
  processedImage: string | null = null;
  private stream: MediaStream | null = null;

  constructor() {
    this.canvas = document.createElement('canvas');
  }

  async ngOnInit() {
    this.socket = io('http://localhost:5000');
    this.setupSocketListeners();
    await this.setupWebcam();
  }

  private setupSocketListeners() {
    this.socket.on('connect', () => console.log('Connected to server'));
    this.socket.on('processed_frame', (data: string) => {
      this.processedImage = data;
    });
    this.socket.on('disconnect', () => console.log('Disconnected from server'));
  }

  async setupWebcam() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      });
      
      const video = this.videoElementRef.nativeElement;
      video.srcObject = this.stream;
      
      video.onloadedmetadata = () => {
        video.play();
        this.startProcessing();
      };

    } catch (err) {
      console.error('Error accessing webcam:', err);
    }
  }

  private startProcessing() {
    const processFrame = () => {
      if (this.socket.connected) {
        this.captureAndSendFrame();
      }
      requestAnimationFrame(processFrame);
    };
    processFrame();
  }

  private captureAndSendFrame() {
    const video = this.videoElementRef.nativeElement;
    this.canvas.width = video.videoWidth;
    this.canvas.height = video.videoHeight;
    
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    this.canvas.toBlob(blob => {
      if (blob) {
        const reader = new FileReader();
        reader.onload = () => {
          const base64data = reader.result as string;
          this.socket.emit('frame', base64data);
        };
        reader.readAsDataURL(blob);
      }
    }, 'image/jpeg', 0.7);
  }

  ngOnDestroy() {
    this.socket.disconnect();
    this.stream?.getTracks().forEach(track => track.stop());
  }
}
