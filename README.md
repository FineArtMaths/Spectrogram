# Spectrogram

This short Python script represents a deliberately naive attempt to extract a coarse spectrogram from a sound. The idea is to use these values directly to drive an additive synthesizer to approximate the sound in a "machine listening" project. 

I did it this way (rather than just using the wonderful [librosa](https://librosa.org)) because I wanted to be able to control all the steps in the process. This is unlikely to be the right approach for *your* project, although it might be...

![spectrograph](https://user-images.githubusercontent.com/5106495/232740842-37ed6ffe-7302-4d7d-b072-412dd6d975b2.png)
