// webcam.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root' // Or 'platform' if using Angular 16+
})
export class WebcamService {
  private apiUrl = 'http://localhost:5000/process';

  constructor(private http: HttpClient) { }

  processFrame(frame: Blob): Observable<Blob> {
    const formData = new FormData();
    formData.append('image', frame);
    return this.http.post(this.apiUrl, formData, { 
      responseType: 'blob',
      headers: { 'Accept': 'image/jpeg' }
    });
  }
}
