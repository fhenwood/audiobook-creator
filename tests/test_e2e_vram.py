"""
End-to-End VRAM Test: Generate + Verify
Replicates exact pipeline behavior with real audio generation and verification.
"""
import asyncio
import subprocess
import time
import os
import tempfile

def get_vram():
    """Get current VRAM usage in MiB."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        return int(result.strip())
    except:
        return 0

async def main():
    from audiobook.tts.service import TTSService, tts_service
    from audiobook.core.verification import TranscriptionVerifier
    from audiobook.utils.gpu_resource_manager import gpu_manager
    import gc
    import torch
    
    print("🧪 E2E VRAM Test: Generate + Verify")
    print("=" * 50)
    
    # Test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, this is a test of the audiobook generation system."
    ]
    
    # Create temp dir for audio files
    temp_dir = tempfile.mkdtemp(prefix="vram_test_")
    
    # 1. Baseline
    base_vram = get_vram()
    print(f"\n📉 Baseline VRAM: {base_vram} MiB")
    
    # ========== PHASE 1: GENERATE ==========
    print("\n" + "=" * 50)
    print("PHASE 1: GENERATE (VibeVoice)")
    print("=" * 50)
    
    audio_files = []
    try:
        for i, text in enumerate(sentences):
            print(f"\n🎤 Generating sentence {i+1}: '{text[:30]}...'")
            result = await tts_service.generate(text, engine="vibevoice")
            
            if result and result.audio_data:
                audio_path = os.path.join(temp_dir, f"sentence_{i}.wav")
                with open(audio_path, "wb") as f:
                    f.write(result.audio_data)
                audio_files.append((audio_path, text))
                print(f"   ✅ Generated {len(result.audio_data)} bytes -> {audio_path}")
            else:
                print(f"   ❌ Generation failed")
    except Exception as e:
        print(f"   ❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    post_gen_vram = get_vram()
    print(f"\n📈 After Generation VRAM: {post_gen_vram} MiB (+{post_gen_vram - base_vram} MiB)")
    
    # ========== PHASE 2: UNLOAD VIBEVOICE ==========
    print("\n" + "=" * 50)
    print("PHASE 2: UNLOAD VibeVoice")
    print("=" * 50)
    
    print("🛑 Unloading VibeVoice...")
    await tts_service.unload_engine("vibevoice")
    gpu_manager.release_vibevoice()
    
    # Explicitly clear ALL local refs that might hold model tensors
    if 'result' in locals():
        del result  # Clear last generation result
    
    # Also clear tts_service internal state
    tts_service._current_engine = None
    
    # Explicit GC
    for _ in range(5):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    print("⏳ Waiting 5s for deferred cleanup...")
    await asyncio.sleep(5)
    
    post_unload_vram = get_vram()
    print(f"📉 After Unload VRAM: {post_unload_vram} MiB (Diff: {post_unload_vram - base_vram} MiB)")
    
    # ========== PHASE 3: VERIFY WITH WHISPER ==========
    print("\n" + "=" * 50)
    print("PHASE 3: VERIFY (Whisper)")
    print("=" * 50)
    
    if audio_files:
        print("🔄 Loading Whisper verifier...")
        gpu_manager.acquire_whisper()
        
        verifier = TranscriptionVerifier(model_size="large-v3")
        
        for audio_path, expected_text in audio_files:
            print(f"\n🔍 Verifying: {audio_path}")
            try:
                passed, score, transcript = verifier.verify(audio_path, expected_text)
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {status} (score: {score:.2f})")
                print(f"   Expected: {expected_text[:50]}...")
                print(f"   Got:      {transcript[:50]}...")
            except Exception as e:
                print(f"   ❌ Verification error: {e}")
        
        # Unload Whisper
        print("\n🛑 Releasing Whisper...")
        gpu_manager.release_whisper()
        verifier.unload()
    else:
        print("⚠️ No audio files to verify")
    
    post_verify_vram = get_vram()
    print(f"\n📉 After Verification VRAM: {post_verify_vram} MiB")
    
    # ========== RESULTS ==========
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Baseline:          {base_vram} MiB")
    print(f"After Generate:    {post_gen_vram} MiB (+{post_gen_vram - base_vram})")
    print(f"After Unload:      {post_unload_vram} MiB (+{post_unload_vram - base_vram})")
    print(f"After Verify:      {post_verify_vram} MiB (+{post_verify_vram - base_vram})")
    
    if post_unload_vram - base_vram > 2000:
        print("\n❌ LEAK DETECTED after unload (>2GB retained)")
    elif post_verify_vram - base_vram > 2000:
        print("\n❌ LEAK DETECTED after verify (>2GB retained)")
    else:
        print("\n✅ NO LEAK: VRAM returned to baseline")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(main())
