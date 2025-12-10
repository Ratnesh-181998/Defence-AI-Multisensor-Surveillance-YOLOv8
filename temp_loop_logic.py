
    # Process frames using a batch loop to reduce UI blinking (re-running script less often)
    # We process 10 frames in a loop before triggering a Streamlit rerun
    for _ in range(10):
        # Check if we should stop (if user clicked stop, it won't register until rerun, 
        # so this loop latency determines stop responsiveness)
        if not st.session_state.stream_active:
            break

        frame_start = time.time()
        
        for idx, cam_name in enumerate(["Day-1", "Day-2", "Thermal-1", "Thermal-2"]):
            # Only process if this camera is selected/active
            if cam_name not in st.session_state.queues:
                continue

            if cam_name in st.session_state.queues:
                try:
                    frame = st.session_state.queues[cam_name].get_nowait()
                except queue.Empty:
                    frame = None
            else:
                frame = None
            
            if frame is None:
                continue # Skip processing if no frame
            
            # Apply enhancement if enabled
            if enable_enhancement:
                try:
                    frame = clahe_enhance(frame)
                except Exception as e:
                    # log(f"Enhancement error: {e}", "ERROR") # Reduce log spam in loop
                    pass
            
            # Run inference (only on first camera or specific logic to save compute?)
            # For demo, run on all active frames
            detections = []
            if st.session_state.engine:
                try:
                    detections = st.session_state.engine.infer(frame)
                except Exception as e:
                    pass
            
            # Apply tracking if enabled
            if enable_tracking and detections:
                try:
                    boxes = [(x, y, w, h) for (x, y, w, h, cls, conf) in detections]
                    tracked_objects = st.session_state.tracker.update(boxes)
                    
                    # Draw detections and tracks
                    for (x, y, w, h, cls, conf) in detections:
                        if conf >= confidence_threshold:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{conf:.2f}", (x, y-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    for obj_id, centroid in tracked_objects.items():
                        cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID:{obj_id}", (centroid[0]+10, centroid[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception as e:
                    pass
            
            # Recording Logic
            if st.session_state.recording_active:
                if cam_name not in st.session_state.recorders:
                    try:
                        record_dir = Path("recordings")
                        record_dir.mkdir(exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = record_dir / f"{cam_name}_{timestamp}.mp4"
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(str(filename), fourcc, 30.0, (w, h))
                        if writer.isOpened():
                            st.session_state.recorders[cam_name] = writer
                    except:
                        pass
                
                if cam_name in st.session_state.recorders:
                    try:
                        st.session_state.recorders[cam_name].write(frame)
                    except:
                        pass
            else:
                # Stop recording
                 if cam_name in st.session_state.recorders:
                    try:
                        st.session_state.recorders[cam_name].release()
                        del st.session_state.recorders[cam_name]
                    except:
                        pass

            # Update Display
            if idx < len(camera_placeholders):
                # Ensure we are not updating a placeholder that doesn't exist
                try:
                   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                   camera_placeholders[idx].image(frame_rgb, caption=cam_name, use_container_width=True)
                except:
                    pass
        
        # FPS Control
        frame_time = time.time() - frame_start
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            st.session_state.fps_history.append(current_fps)
        
        # Sleep to maintain target FPS
        time.sleep(max(0.01, 1.0 / fps_target))

    # After batch loop, rerun to handle events
    st.rerun()
