import logging
import os
import torchaudio
import soundfile as sf
import dtlpy as dl
import tempfile
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(name='dataset-pytorch')


class DatasetPytorch(dl.BaseServiceRunner):
    """
    A class for loading and processing the LibriSpeech dataset using PyTorch.
    It includes methods for downloading the dataset, saving audio and transcriptions,
    and uploading the dataset to the DataLoop platform.
    """

    def __init__(self):
        """
        Initializes the PyTorch dataset loader, setting up the directory structure
        and downloading the LibriSpeech dataset.
        """
        self.logger = logger
        self.logger.info('Initializing PyTorch dataset loader')
        data_dir_file = 'librispeech'
        self.data_dir = os.path.join(os.getcwd(), data_dir_file)
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset_part = 'dev-clean'

        self.dataset_pytorch = torchaudio.datasets.LIBRISPEECH(root=self.data_dir, url=self.dataset_part, download=True)
        self.logger.info('Dataset loaded')

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        """
        Prepares and uploads the dataset to the Dataloop platform.

        :param dataset: The Dataloop dataset object where the data will be uploaded.
        :param source: The source of the dataset, used for logging purposes.
        """
        temp_dir = tempfile.TemporaryDirectory()
        pool = ThreadPool(processes=5)
        async_results = list()
        self.logger.info('Uploading dataset')
        ranges = 1000

        for i in range(ranges):
            audio, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset_pytorch[i]
            audio_filename = f"{speaker_id}_{chapter_id}_{utterance_id}.wav"
            # Length of the audio data
            num_samples = audio.size(1)
            # Duration in seconds
            duration_seconds = num_samples / sample_rate
            audio_filename = f"{speaker_id}_{chapter_id}_{utterance_id}.wav"
            full_audio_path = os.path.join(temp_dir.name, audio_filename)
            torchaudio.save(full_audio_path, audio, sample_rate)
            async_results.append(
                pool.apply_async(
                    self.upload_item_with_annotations,
                    kwds={
                        "audio_path": full_audio_path,
                        "dataset": dataset,
                        "annotation_text": transcript,
                        "label": speaker_id,
                        "end_time": duration_seconds
                    },
                )
            )

        pool.close()
        pool.join()

    @staticmethod
    def upload_item_with_annotations(audio_path: str, dataset: dl.Dataset, annotation_text: str, label: str, end_time: float):
        """
        Uploads an audio item with annotations to the Dataloop platform.

        :param audio_path: The path to the audio file.
        :param dataset: The Dataloop dataset object where the data will be uploaded.
        :param annotation_text: The transcription of the audio.
        :param label: The speaker ID.
        :param end_time: The duration of the audio.
        """
        # Upload audio file
        item = dataset.items.upload(local_path=audio_path, remote_path='/')
        builder = item.annotations.builder()
        annotation_definition = dl.Subtitle(text=annotation_text, label=str(label))
        builder.add(annotation_definition=annotation_definition,
                    start_time=0,
                    end_time=end_time,
                    object_id='001'
                    )
        item.annotations.upload(builder)
        return item
