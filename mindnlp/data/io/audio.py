# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=superfluous-parens
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string
# pylint: disable=too-many-return-statements
"""audio io"""
import collections
import io
import struct
import sys
import warnings
from enum import IntEnum

import numpy as np

__all__ = [
    "read",
    "write",
]


class WaveFormat(IntEnum):
    """
    WAVE form wFormatTag IDs

    Complete list is in mmreg.h in Windows 10 SDK.  ALAC and OPUS are the
    newest additions, in v10.0.14393 2016-07
    """

    UNKNOWN = 0x0000  # Microsoft Corporation
    PCM = 0x0001  # Microsoft Corporation
    ADPCM = 0x0002  # Microsoft Corporation
    IEEE_FLOAT = 0x0003  # Microsoft Corporation
    VSELP = 0x0004  # Compaq Computer Corp.
    IBM_CVSD = 0x0005  # IBM Corporation
    ALAW = 0x0006  # Microsoft Corporation
    MULAW = 0x0007  # Microsoft Corporation
    DTS = 0x0008  # Microsoft Corporation
    DRM = 0x0009  # Microsoft Corporation
    WMAVOICE9 = 0x000A  # Microsoft Corporation
    WMAVOICE10 = 0x000B  # Microsoft Corporation

    OKI_ADPCM = 0x0010  # OKI
    DVI_ADPCM = 0x0011  # Intel Corporation
    IMA_ADPCM = DVI_ADPCM  # Intel Corporation
    MEDIASPACE_ADPCM = 0x0012  # Videologic
    SIERRA_ADPCM = 0x0013  # Sierra Semiconductor Corp
    G723_ADPCM = 0x0014  # Antex Electronics Corporation
    DIGISTD = 0x0015  # DSP Solutions, Inc.
    DIGIFIX = 0x0016  # DSP Solutions, Inc.
    DIALOGIC_OKI_ADPCM = 0x0017  # Dialogic Corporation
    MEDIAVISION_ADPCM = 0x0018  # Media Vision, Inc.
    CU_CODEC = 0x0019  # Hewlett-Packard Company
    HP_DYN_VOICE = 0x001A  # Hewlett-Packard Company

    YAMAHA_ADPCM = 0x0020  # Yamaha Corporation of America
    SONARC = 0x0021  # Speech Compression
    DSPGROUP_TRUESPEECH = 0x0022  # DSP Group, Inc
    ECHOSC1 = 0x0023  # Echo Speech Corporation
    AUDIOFILE_AF36 = 0x0024  # Virtual Music, Inc.
    APTX = 0x0025  # Audio Processing Technology
    AUDIOFILE_AF10 = 0x0026  # Virtual Music, Inc.
    PROSODY_1612 = 0x0027  # Aculab plc
    LRC = 0x0028  # Merging Technologies S.A.

    DOLBY_AC2 = 0x0030  # Dolby Laboratories
    GSM610 = 0x0031  # Microsoft Corporation
    MSNAUDIO = 0x0032  # Microsoft Corporation
    ANTEX_ADPCME = 0x0033  # Antex Electronics Corporation
    CONTROL_RES_VQLPC = 0x0034  # Control Resources Limited
    DIGIREAL = 0x0035  # DSP Solutions, Inc.
    DIGIADPCM = 0x0036  # DSP Solutions, Inc.
    CONTROL_RES_CR10 = 0x0037  # Control Resources Limited
    NMS_VBXADPCM = 0x0038  # Natural MicroSystems
    CS_IMAADPCM = 0x0039  # Crystal Semiconductor IMA ADPCM
    ECHOSC3 = 0x003A  # Echo Speech Corporation
    ROCKWELL_ADPCM = 0x003B  # Rockwell International
    ROCKWELL_DIGITALK = 0x003C  # Rockwell International
    XEBEC = 0x003D  # Xebec Multimedia Solutions Limited

    G721_ADPCM = 0x0040  # Antex Electronics Corporation
    G728_CELP = 0x0041  # Antex Electronics Corporation
    MSG723 = 0x0042  # Microsoft Corporation
    INTEL_G723_1 = 0x0043  # Intel Corp.
    INTEL_G729 = 0x0044  # Intel Corp.
    SHARP_G726 = 0x0045  # Sharp

    MPEG = 0x0050  # Microsoft Corporation
    RT24 = 0x0052  # InSoft, Inc.
    PAC = 0x0053  # InSoft, Inc.
    MPEGLAYER3 = 0x0055  # ISO/MPEG Layer3 Format Tag
    LUCENT_G723 = 0x0059  # Lucent Technologies

    CIRRUS = 0x0060  # Cirrus Logic
    ESPCM = 0x0061  # ESS Technology
    VOXWARE = 0x0062  # Voxware Inc
    CANOPUS_ATRAC = 0x0063  # Canopus, co., Ltd.
    G726_ADPCM = 0x0064  # APICOM
    G722_ADPCM = 0x0065  # APICOM
    DSAT = 0x0066  # Microsoft Corporation
    DSAT_DISPLAY = 0x0067  # Microsoft Corporation
    VOXWARE_BYTE_ALIGNED = 0x0069  # Voxware Inc

    VOXWARE_AC8 = 0x0070  # Voxware Inc
    VOXWARE_AC10 = 0x0071  # Voxware Inc
    VOXWARE_AC16 = 0x0072  # Voxware Inc
    VOXWARE_AC20 = 0x0073  # Voxware Inc
    VOXWARE_RT24 = 0x0074  # Voxware Inc
    VOXWARE_RT29 = 0x0075  # Voxware Inc
    VOXWARE_RT29HW = 0x0076  # Voxware Inc
    VOXWARE_VR12 = 0x0077  # Voxware Inc
    VOXWARE_VR18 = 0x0078  # Voxware Inc
    VOXWARE_TQ40 = 0x0079  # Voxware Inc
    VOXWARE_SC3 = 0x007A  # Voxware Inc
    VOXWARE_SC3_1 = 0x007B  # Voxware Inc

    SOFTSOUND = 0x0080  # Softsound, Ltd.
    VOXWARE_TQ60 = 0x0081  # Voxware Inc
    MSRT24 = 0x0082  # Microsoft Corporation
    G729A = 0x0083  # AT&T Labs, Inc.
    MVI_MVI2 = 0x0084  # Motion Pixels
    DF_G726 = 0x0085  # DataFusion Systems (Pty) (Ltd)
    DF_GSM610 = 0x0086  # DataFusion Systems (Pty) (Ltd)
    ISIAUDIO = 0x0088  # Iterated Systems, Inc.
    ONLIVE = 0x0089  # * OnLive! Technologies, Inc.
    MULTITUDE_FT_SX20 = 0x008A  # Multitude Inc.
    INFOCOM_ITS_G721_ADPCM = 0x008B  # Infocom
    CONVEDIA_G729 = 0x008C  # Convedia Corp.
    CONGRUENCY = 0x008D  # Congruency Inc.

    SBC24 = 0x0091  # Siemens Business Communications Sys
    DOLBY_AC3_SPDIF = 0x0092  # Sonic Foundry
    MEDIASONIC_G723 = 0x0093  # MediaSonic
    PROSODY_8KBPS = 0x0094  # Aculab plc
    ZYXEL_ADPCM = 0x0097  # ZyXEL Communications, Inc.
    PHILIPS_LPCBB = 0x0098  # Philips Speech Processing
    PACKED = 0x0099  # Studer Professional Audio AG

    MALDEN_PHONYTALK = 0x00A0  # Malden Electronics Ltd.
    RACAL_RECORDER_GSM = 0x00A1  # Racal recorders
    RACAL_RECORDER_G720_A = 0x00A2  # Racal recorders
    RACAL_RECORDER_G723_1 = 0x00A3  # Racal recorders
    RACAL_RECORDER_TETRA_ACELP = 0x00A4  # Racal recorders

    NEC_AAC = 0x00B0  # NEC Corp.
    # For Raw AAC, with format block AudioSpecificConfig()
    # (as defined by MPEG-4), that follows WAVEFORMATEX
    RAW_AAC1 = 0x00FF

    RHETOREX_ADPCM = 0x0100  # Rhetorex Inc.
    IRAT = 0x0101  # BeCubed Software Inc.
    VIVO_G723 = 0x0111  # Vivo Software
    VIVO_SIREN = 0x0112  # Vivo Software

    PHILIPS_CELP = 0x0120  # Philips Speech Processing
    PHILIPS_GRUNDIG = 0x0121  # Philips Speech Processing
    DIGITAL_G723 = 0x0123  # Digital Equipment Corporation
    SANYO_LD_ADPCM = 0x0125  # Sanyo Electric Co., Ltd.

    SIPROLAB_ACEPLNET = 0x0130  # Sipro Lab Telecom Inc.
    SIPROLAB_ACELP4800 = 0x0131  # Sipro Lab Telecom Inc.
    SIPROLAB_ACELP8V3 = 0x0132  # Sipro Lab Telecom Inc.
    SIPROLAB_G729 = 0x0133  # Sipro Lab Telecom Inc.
    SIPROLAB_G729A = 0x0134  # Sipro Lab Telecom Inc.
    SIPROLAB_KELVIN = 0x0135  # Sipro Lab Telecom Inc.
    VOICEAGE_AMR = 0x0136  # VoiceAge Corp.

    G726ADPCM = 0x0140  # Dictaphone Corporation
    DICTAPHONE_CELP68 = 0x0141  # Dictaphone Corporation
    DICTAPHONE_CELP54 = 0x0142  # Dictaphone Corporation

    QUALCOMM_PUREVOICE = 0x0150  # Qualcomm, Inc.
    QUALCOMM_HALFRATE = 0x0151  # Qualcomm, Inc.
    TUBGSM = 0x0155  # Ring Zero Systems, Inc.

    MSAUDIO1 = 0x0160  # Microsoft Corporation
    WMAUDIO2 = 0x0161  # Microsoft Corporation
    WMAUDIO3 = 0x0162  # Microsoft Corporation
    WMAUDIO_LOSSLESS = 0x0163  # Microsoft Corporation
    WMASPDIF = 0x0164  # Microsoft Corporation

    UNISYS_NAP_ADPCM = 0x0170  # Unisys Corp.
    UNISYS_NAP_ULAW = 0x0171  # Unisys Corp.
    UNISYS_NAP_ALAW = 0x0172  # Unisys Corp.
    UNISYS_NAP_16K = 0x0173  # Unisys Corp.
    SYCOM_ACM_SYC008 = 0x0174  # SyCom Technologies
    SYCOM_ACM_SYC701_G726L = 0x0175  # SyCom Technologies
    SYCOM_ACM_SYC701_CELP54 = 0x0176  # SyCom Technologies
    SYCOM_ACM_SYC701_CELP68 = 0x0177  # SyCom Technologies
    KNOWLEDGE_ADVENTURE_ADPCM = 0x0178  # Knowledge Adventure, Inc.

    FRAUNHOFER_IIS_MPEG2_AAC = 0x0180  # Fraunhofer IIS
    DTS_DS = 0x0190  # * Digital Theatre Systems, Inc.

    CREATIVE_ADPCM = 0x0200  # Creative Labs, Inc
    CREATIVE_FASTSPEECH8 = 0x0202  # Creative Labs, Inc
    CREATIVE_FASTSPEECH10 = 0x0203  # Creative Labs, Inc
    UHER_ADPCM = 0x0210  # UHER informatic GmbH
    ULEAD_DV_AUDIO = 0x0215  # Ulead Systems, Inc.
    ULEAD_DV_AUDIO_1 = 0x0216  # Ulead Systems, Inc.

    QUARTERDECK = 0x0220  # Quarterdeck Corporation
    ILINK_VC = 0x0230  # I-link Worldwide
    RAW_SPORT = 0x0240  # Aureal Semiconductor
    ESST_AC3 = 0x0241  # ESS Technology, Inc.
    GENERIC_PASSTHRU = 0x0249
    IPI_HSX = 0x0250  # Interactive Products, Inc.
    IPI_RPELP = 0x0251  # Interactive Products, Inc.

    CS2 = 0x0260  # Consistent Software
    SONY_SCX = 0x0270  # Sony Corp.
    SONY_SCY = 0x0271  # Sony Corp.
    SONY_ATRAC3 = 0x0272  # Sony Corp.
    SONY_SPC = 0x0273  # Sony Corp.
    TELUM_AUDIO = 0x0280  # Telum Inc.
    TELUM_IA_AUDIO = 0x0281  # Telum Inc.
    NORCOM_VOICE_SYSTEMS_ADPCM = 0x0285  # Norcom Electronics Corp.

    FM_TOWNS_SND = 0x0300  # Fujitsu Corp.
    MICRONAS = 0x0350  # Micronas Semiconductors, Inc.
    MICRONAS_CELP833 = 0x0351  # Micronas Semiconductors, Inc.
    BTV_DIGITAL = 0x0400  # Brooktree Corporation
    INTEL_MUSIC_CODER = 0x0401  # Intel Corp.
    INDEO_AUDIO = 0x0402  # Ligos
    QDESIGN_MUSIC = 0x0450  # QDesign Corporation
    ON2_VP7_AUDIO = 0x0500  # On2 Technologies
    ON2_VP6_AUDIO = 0x0501  # On2 Technologies
    VME_VMPCM = 0x0680  # AT&T Labs, Inc.
    TPC = 0x0681  # AT&T Labs, Inc.
    LIGHTWAVE_LOSSLESS = 0x08AE  # Clearjump

    OLIGSM = 0x1000  # Ing C. Olivetti & C., S.p.A.
    OLIADPCM = 0x1001  # Ing C. Olivetti & C., S.p.A.
    OLICELP = 0x1002  # Ing C. Olivetti & C., S.p.A.
    OLISBC = 0x1003  # Ing C. Olivetti & C., S.p.A.
    OLIOPR = 0x1004  # Ing C. Olivetti & C., S.p.A.
    LH_CODEC = 0x1100  # Lernout & Hauspie
    LH_CODEC_CELP = 0x1101  # Lernout & Hauspie
    LH_CODEC_SBC8 = 0x1102  # Lernout & Hauspie
    LH_CODEC_SBC12 = 0x1103  # Lernout & Hauspie
    LH_CODEC_SBC16 = 0x1104  # Lernout & Hauspie
    NORRIS = 0x1400  # Norris Communications, Inc.
    ISIAUDIO_2 = 0x1401  # ISIAudio
    SOUNDSPACE_MUSICOMPRESS = 0x1500  # AT&T Labs, Inc.
    MPEG_ADTS_AAC = 0x1600  # Microsoft Corporation
    MPEG_RAW_AAC = 0x1601  # Microsoft Corporation
    # MPEG-4 Audio Transport Streams (LOAS/LATM)
    MPEG_LOAS = 0x1602  # Microsoft Corporation

    NOKIA_MPEG_ADTS_AAC = 0x1608  # Microsoft Corporation
    NOKIA_MPEG_RAW_AAC = 0x1609  # Microsoft Corporation
    VODAFONE_MPEG_ADTS_AAC = 0x160A  # Microsoft Corporation
    VODAFONE_MPEG_RAW_AAC = 0x160B  # Microsoft Corporation
    # MPEG-2 AAC or MPEG-4 HE-AAC v1/v2 streams
    # with any payload (ADTS, ADIF, LOAS/LATM, RAW).
    # Format block includes MP4 AudioSpecificConfig()  -- see HEAACWAVEFORMAT below
    MPEG_HEAAC = 0x1610  # Microsoft Corporation

    VOXWARE_RT24_SPEECH = 0x181C  # Voxware Inc.
    SONICFOUNDRY_LOSSLESS = 0x1971  # Sonic Foundry
    INNINGS_TELECOM_ADPCM = 0x1979  # Innings Telecom Inc.
    LUCENT_SX8300P = 0x1C07  # Lucent Technologies
    LUCENT_SX5363S = 0x1C0C  # Lucent Technologies
    CUSEEME = 0x1F03  # CUSeeMe
    NTCSOFT_ALF2CM_ACM = 0x1FC4  # NTCSoft

    DVM = 0x2000  # FAST Multimedia AG
    DTS2 = 0x2001
    MAKEAVIS = 0x3313
    DIVIO_MPEG4_AAC = 0x4143  # Divio, Inc.
    NOKIA_ADAPTIVE_MULTIRATE = 0x4201  # Nokia
    DIVIO_G726 = 0x4243  # Divio, Inc.
    LEAD_SPEECH = 0x434C  # LEAD Technologies
    LEAD_VORBIS = 0x564C  # LEAD Technologies
    WAVPACK_AUDIO = 0x5756  # xiph.org

    OGG_VORBIS_MODE_1 = 0x674F  # Ogg Vorbis
    OGG_VORBIS_MODE_2 = 0x6750  # Ogg Vorbis
    OGG_VORBIS_MODE_3 = 0x6751  # Ogg Vorbis
    OGG_VORBIS_MODE_1_PLUS = 0x676F  # Ogg Vorbis
    OGG_VORBIS_MODE_2_PLUS = 0x6770  # Ogg Vorbis
    OGG_VORBIS_MODE_3_PLUS = 0x6771  # Ogg Vorbis
    ALAC = 0x6C61  # Apple Lossless

    # Can't have leading digit
    _3COM_NBX = 0x7000  # 3COM Corp.

    OPUS = 0x704F  # Opus
    FAAD_AAC = 0x706D
    AMR_NB = 0x7361  # AMR Narrowband
    AMR_WB = 0x7362  # AMR Wideband
    AMR_WP = 0x7363  # AMR Wideband Plus
    GSM_AMR_CBR = 0x7A21  # GSMA/3GPP
    GSM_AMR_VBR_SID = 0x7A22  # GSMA/3GPP

    COMVERSE_INFOSYS_G723_1 = 0xA100  # Comverse Infosys
    COMVERSE_INFOSYS_AVQSBC = 0xA101  # Comverse Infosys
    COMVERSE_INFOSYS_SBC = 0xA102  # Comverse Infosys
    SYMBOL_G729_A = 0xA103  # Symbol Technologies
    VOICEAGE_AMR_WB = 0xA104  # VoiceAge Corp.
    INGENIENT_G726 = 0xA105  # Ingenient Technologies, Inc.
    MPEG4_AAC = 0xA106  # ISO/MPEG-4
    ENCORE_G726 = 0xA107  # Encore Software
    ZOLL_ASAO = 0xA108  # ZOLL Medical Corp.
    SPEEX_VOICE = 0xA109  # xiph.org
    VIANIX_MASC = 0xA10A  # Vianix LLC
    WM9_SPECTRUM_ANALYZER = 0xA10B  # Microsoft
    WMF_SPECTRUM_ANAYZER = 0xA10C  # Microsoft
    GSM_610 = 0xA10D
    GSM_620 = 0xA10E
    GSM_660 = 0xA10F

    GSM_690 = 0xA110
    GSM_ADAPTIVE_MULTIRATE_WB = 0xA111
    POLYCOM_G722 = 0xA112  # Polycom
    POLYCOM_G728 = 0xA113  # Polycom
    POLYCOM_G729_A = 0xA114  # Polycom
    POLYCOM_SIREN = 0xA115  # Polycom
    GLOBAL_IP_ILBC = 0xA116  # Global IP
    RADIOTIME_TIME_SHIFT_RADIO = 0xA117  # RadioTime
    NICE_ACA = 0xA118  # Nice Systems
    NICE_ADPCM = 0xA119  # Nice Systems
    VOCORD_G721 = 0xA11A  # Vocord Telecom
    VOCORD_G726 = 0xA11B  # Vocord Telecom
    VOCORD_G722_1 = 0xA11C  # Vocord Telecom
    VOCORD_G728 = 0xA11D  # Vocord Telecom
    VOCORD_G729 = 0xA11E  # Vocord Telecom
    VOCORD_G729_A = 0xA11F  # Vocord Telecom

    VOCORD_G723_1 = 0xA120  # Vocord Telecom
    VOCORD_LBC = 0xA121  # Vocord Telecom
    NICE_G728 = 0xA122  # Nice Systems
    FRACE_TELECOM_G729 = 0xA123  # France Telecom
    CODIAN = 0xA124  # CODIAN

    FLAC = 0xF1AC  # flac.sourceforge.net
    EXTENSIBLE = 0xFFFE  # Microsoft
    DEVELOPMENT = 0xFFFF


# will be supported: WaveFormat.ALAW, WaveFormat.MULAW, WaveFormat.EXTENSIBLE
SUPPORTED_WAVE_FORMATS = {WaveFormat.PCM, WaveFormat.IEEE_FLOAT}


class WavFileWarning(UserWarning):
    pass


class Endian:
    big_endian = "big endian"
    small_endian = "small endian"


def _fmt_chunk(file_to_read, endian):
    if endian == Endian.big_endian:
        fmt = ">"
    else:
        fmt = "<"
    chunk = file_to_read.read(4)
    chunk_size = struct.unpack(f"{fmt}I", chunk)[0]  # usually: 16 or 18 or 40
    if chunk_size < 16:
        raise ValueError("Binary structure of wave file is not compliant")

    # fmt chunk data
    bytes_read = 16
    res = struct.unpack(f"{fmt}HHIIHH", file_to_read.read(bytes_read))
    format_tag, channels, samplerate, bytes_per_second, block_align, bit_depth = res

    if format_tag == WaveFormat.EXTENSIBLE and chunk_size >= (bytes_read + 2):
        # usually: 0 or 22
        ext_chunk_size = struct.unpack(f"{fmt}H", file_to_read.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = file_to_read.read(22)
            bytes_read += 22
            # valid_bits_per_sample = extensible_chunk_data[:2]
            # channel_mask = extensible_chunk_data[2 : 2 + 4]
            raw_guid = extensible_chunk_data[2 + 4 : 2 + 4 + 16]

            # GUID template {XXXXXXXX-0000-0010-8000-00AA00389B71} (RFC-2361)
            # MS GUID byte order: first three groups are native byte order,
            # rest is Big Endian
            if endian == Endian.big_endian:
                tail = b"\x00\x00\x00\x10\x80\x00\x00\xAA\x00\x38\x9B\x71"
            else:  # small_endian
                tail = b"\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71"
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(f"{fmt}I", raw_guid[:4])[0]

        else:
            raise ValueError("Binary structure of wave file is not compliant")

    if format_tag not in SUPPORTED_WAVE_FORMATS:
        try:
            format_name = WaveFormat(format_tag).name
        except ValueError:
            format_name = f"{format_tag:#06x}"
        raise ValueError(
            f"Unknown wave file format: {format_name}. Supported "
            "formats: " + ", ".join(x.name for x in SUPPORTED_WAVE_FORMATS)
        )

    # move file pointer to next chunk
    if chunk_size > bytes_read:
        file_to_read.read(chunk_size - bytes_read)

    # fmt should always be 16, 18 or 40, but handle it just in case
    # "If the chunk size is an odd number of bytes, a pad byte with value zero
    # is written after ckData." So we need to seek past this after each chunk.
    if chunk_size % 2:
        file_to_read.seek(1, 1)

    if format_tag == WaveFormat.PCM:
        if bytes_per_second != samplerate * block_align:
            raise ValueError(
                "WAV header is invalid: nAvgBytesPerSec must"
                " equal product of nSamplesPerSec and"
                " nBlockAlign, but file has nSamplesPerSec ="
                f" {samplerate}, nBlockAlign = {block_align}, and"
                f" nAvgBytesPerSec = {bytes_per_second}"
            )

    return (
        format_tag,
        channels,
        samplerate,
        bytes_per_second,
        block_align,
        bit_depth,
    )


def _data_chunk(
    file_to_read,
    format_tag,
    channels,
    bit_depth,
    endian,
    samplerate,
    block_align,
    offset,
    duration,
):
    if endian == Endian.big_endian:
        fmt = ">"
    else:
        fmt = "<"

    # Size of the data subchunk in bytes
    size = struct.unpack(fmt + "I", file_to_read.read(4))[0]
    # Number of bytes per sample (sample container size)
    bytes_per_sample = block_align // channels
    all_samples = size // bytes_per_sample
    n_samples = all_samples

    if format_tag == WaveFormat.PCM:
        if 1 <= bit_depth <= 8:
            dtype = "u1"  # WAV of 8-bit integer or less are unsigned
        elif bytes_per_sample in {3, 5, 6, 7}:
            # No compatible dtype.  Load as raw bytes for reshaping later.
            dtype = "V1"
        elif bit_depth <= 64:
            # Remaining bit depths can map directly to signed numpy dtypes
            dtype = f"{fmt}i{bytes_per_sample}"
        else:
            raise ValueError(
                "Unsupported bit depth: the WAV file "
                f"has {bit_depth}-bit integer data."
            )
    elif format_tag == WaveFormat.IEEE_FLOAT:
        if bit_depth in {32, 64}:
            dtype = f"{fmt}f{bytes_per_sample}"
            # if bit_depth == 32:
            #     dtype = f'{fmt}nf'
            # else:
            #     dtype = f'{fmt}nd'
        else:
            raise ValueError(
                "Unsupported bit depth: the WAV file "
                f"has {bit_depth}-bit floating-point data."
            )
    else:
        try:
            format_name = WaveFormat(format_tag).name
        except ValueError:
            format_name = f"{format_tag:#06x}"
        raise ValueError(
            f"Unknown wave file format: {format_name}. Supported "
            "formats: " + ", ".join(x.name for x in SUPPORTED_WAVE_FORMATS)
        )
    start = file_to_read.tell()

    ignore_samples = 0
    if offset > 0:
        ignore_samples = int(offset * samplerate)
        file_to_read.read(ignore_samples)
        start = file_to_read.tell()

    try:
        count = size if dtype == "V1" else n_samples
        if ignore_samples <= count:
            count = count - ignore_samples
        if duration and duration * samplerate < count:
            count = int(duration * samplerate)
        # data = struct.unpack(
        #     dtype.replace('n', str(count // bytes_per_sample)),
        #     file_to_read.read(count - count % 4)
        # )
        data = np.fromfile(file_to_read, dtype=dtype, count=count)
        # if count % 4:
        #     pad_byte = file_to_read.read(count % 4)

    except io.UnsupportedOperation:  # not a C-like file
        # just in case it seeked, though it shouldn't
        file_to_read.seek(start, 0)
        data = np.frombuffer(file_to_read.read(size), dtype=dtype)

    if dtype == "V1":
        # Rearrange raw bytes into smallest compatible numpy dtype
        dt = f"{fmt}i4" if bytes_per_sample == 3 else f"{fmt}i8"
        a = np.zeros((len(data) // bytes_per_sample, np.dtype(dt).itemsize), dtype="V1")
        if endian == Endian.big_endian:
            a[:, :bytes_per_sample] = data.reshape((-1, bytes_per_sample))
        else:
            a[:, -bytes_per_sample:] = data.reshape((-1, bytes_per_sample))
        data = a.view(dt).reshape(a.shape[:-1])

    if size % 2:
        file_to_read.seek(1, 1)

    data = np.array(data)
    if channels > 1:
        data = data.reshape(-1, channels)
    return data


def _skip_unknown_chunk(file_to_read, endian):
    if endian == Endian.big_endian:
        fmt = ">I"
    else:
        fmt = "<I"

    data = file_to_read.read(4)
    # call unpack() and seek() only if we have really read data from file
    # otherwise empty read at the end of the file would trigger
    # unnecessary exception at unpack() call
    # in case data equals somehow to 0, there is no need for seek() anyway
    if data:
        size = struct.unpack(fmt, data)[0]
        file_to_read.seek(size, 1)
        # "If the chunk size is an odd number of bytes, a pad byte with value zero
        # is written after ckData."
        # So we need to seek past this after each chunk.
        if size % 2:
            file_to_read.seek(1, 1)


def read(file, offset=0.0, duration=None):
    """
    Open a WAV file.
    Return data and the sample rate
    (in samples/sec) from an LPCM WAV file.

    Args
    ----------
    file : string or open file handle
        Input WAV file.
    offset : float
        start reading after this time (in seconds)
    duration : float
        only load up to this much audio (in seconds)

    Returns
    -------
    audio : np.ndarray
        Data read from WAV file. Data-type is determined from the file;
        see Notes.  Data is 1-D for 1-channel WAV, or 2-D of shape
        (Nsamples, Nchannels) otherwise. If a file-like input without a
        C-like file descriptor (e.g., :class:`python:io.BytesIO`) is
        passed, this will not be writeable. To
    samplerate : int
        Sample rate of WAV file.

    Notes
    -----
    Common data types: [1]_
    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit integer PCM     -2147483648  +2147483647  int32
    24-bit integer PCM     -2147483648  +2147483392  int32
    16-bit integer PCM     -32768       +32767       int16
    8-bit integer PCM      0            255          uint8
    =====================  ===========  ===========  =============
    WAV files can specify arbitrary bit depth, and this function supports
    reading any integer PCM depth from 1 to 64 bits.  Data is returned in the
    smallest compatible numpy int type, in left-justified format.  8-bit and
    lower is unsigned, while 9-bit and higher is signed.
    For example, 24-bit data will be stored as int32, with the MSB of the
    24-bit data stored at the MSB of the int32, and typically the least
    significant byte is 0x00.  (However, if a file actually contains data past
    its specified bit depth, those bits will be read and output, too. [2]_)
    This bit justification and sign matches WAV's native internal format, which
    allows memory mapping of WAV files that use 1, 2, 4, or 8 bytes per sample
    (so 24-bit files cannot be memory-mapped, but 32-bit can).
    IEEE float PCM in 32- or 64-bit format is supported, with or without mmap.
    Values exceeding [-1, +1] are not clipped.
    Non-linear PCM (mu-law, A-law) is not supported.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html
    .. [2] Adobe Systems Incorporated, "Adobe Audition 3 User Guide", section
       "Audio file formats: 24-bit Packed Int (type 1, 20-bit)", 2007

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> import mindspore.dataset.audio

    Get a multi-channel audio file from the tests/data directory.
    >>> data_dir = pjoin(dirname(mindspore.dataset.audio.__file__), 'tests', 'data')
    >>> wav_fname = pjoin(data_dir, 'test-44100Hz-2ch-32bit-float-be.wav')

    Load the .wav file contents.
    >>> audio, sr = read(wav_fname)
    >>> print(f"number of channels = {audio.shape[1]}")
    number of channels = 2
    >>> length = audio.shape[0] / sr
    >>> print(f"length = {length}s")
    length = 0.01s

    Plot the waveform.
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time = np.linspace(0., length, audio.shape[0])
    >>> plt.plot(time, audio[:, 0], label="Left channel")
    >>> plt.plot(time, audio[:, 1], label="Right channel")
    >>> plt.legend()
    >>> plt.xlabel("Time [s]")
    >>> plt.ylabel("Amplitude")
    >>> plt.show()

    """
    if hasattr(file, "read"):
        file_to_read = file
    else:
        file_to_read = open(file, "rb")

    try:
        # ------riff chunk------
        # [0, 4) chunk id
        str1 = file_to_read.read(4)
        if str1 == b"RIFF":
            endian = Endian.small_endian
            fmt = "<"
        elif str1 == b"RIFX":
            endian = Endian.big_endian
            fmt = ">"
        else:
            # There are also .wav files with "FFIR" or "XFIR" signatures?
            raise ValueError(
                f"File format {repr(str1)} not understood. Only "
                "'RIFF' and 'RIFX' supported."
            )

        # [4, 8) chunk size
        str2 = file_to_read.read(4)
        # Size of entire file
        file_size = struct.unpack(f"{fmt}I", str2)[0] + 8

        # [8, 12) type
        str3 = file_to_read.read(4)
        if str3 != b"WAVE":
            raise (f"Not a WAV file. RIFF form type is {repr(str3)}.")

        fmt_chunk_received = False
        data_chunk_received = False
        while file_to_read.tell() < file_size:
            chunk = file_to_read.read(4)
            if not chunk:
                if data_chunk_received:
                    # End of file but data successfully read
                    warnings.warn(
                        "Reached EOF prematurely; finished at {:d} bytes, "
                        "expected {:d} bytes from header.".format(
                            file_to_read.tell(), file_size
                        ),
                        WavFileWarning,
                        stacklevel=2,
                    )
                    break
                raise ValueError("Unexpected end of file.")
            if len(chunk) < 4:
                msg = f"Incomplete chunk ID: {repr(chunk)}"
                # If we have the data, ignore the broken chunk
                if fmt_chunk_received and data_chunk_received:
                    warnings.warn(msg + ", ignoring it.", WavFileWarning, stacklevel=2)
                else:
                    raise ValueError(msg)

            if chunk == b"fmt ":
                fmt_chunk_received = True
                fmt_chunk_info = _fmt_chunk(file_to_read, endian)
                format_tag, channels, samplerate = fmt_chunk_info[0:3]
                _, block_align, bit_depth = fmt_chunk_info[3:6]

            elif chunk == b"data":
                data_chunk_received = True
                if not fmt_chunk_received:
                    raise ValueError("No fmt chunk before data")
                audio = _data_chunk(
                    file_to_read,
                    format_tag,
                    channels,
                    bit_depth,
                    endian,
                    samplerate,
                    block_align,
                    offset,
                    duration,
                )

            elif chunk in {b"fact", b"LIST", b"JUNK", b"Fake"}:
                # Skip alignment chunks without warning
                _skip_unknown_chunk(file_to_read, endian)
            else:
                warnings.warn(
                    "Chunk (non-data) not understood, skipping it.",
                    WavFileWarning,
                    stacklevel=2,
                )
                _skip_unknown_chunk(file_to_read, endian)

    finally:
        if not hasattr(file, "read"):
            file_to_read.close()
        else:
            file_to_read.seek(0)

    # Unified output format
    audiodtype = audio.dtype
    if audiodtype == "int32":
        audio = audio / 2147483648
    elif audiodtype == "int16":
        audio = audio / 32768

    return audio, samplerate


def write(file, data, sr):
    """
    Write a numpy array as a WAV file.

    Args
    ----------
    file : string or open file handle
        Output wav file.
    data : np.ndarray
        A 1-D or 2-D numpy array of either integer or float data-type.
    sr : int
        The sample rate (in samples/sec).

    Notes
    -----
    * Writes a simple uncompressed WAV file.
    * To write multiple-channels, use a 2-D array of shape
      (Nsamples, Nchannels).
    * The bits-per-sample and PCM/float will be determined by the data-type.
    Common data types: [1]_
    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============
    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    Examples
    --------
    Create a 100Hz sine wave, sampled at 44100Hz.
    Write to 16-bit PCM, Mono.
    >>> samplerate = 44100
    >>> fs = 100
    >>> t = np.linspace(0., 1., samplerate)
    >>> amplitude = np.iinfo(np.int16).max
    >>> data = amplitude * np.sin(2. * np.pi * fs * t)
    >>> write("example.wav", data, samplerate)

    """
    if hasattr(file, "write"):
        fid = file
    else:
        fid = open(file, "wb")

    fs = sr

    if data.dtype in ('float64', 'float16'):
        data = data.astype(np.float32)

    try:
        dkind = data.dtype.kind
        if not (
            dkind == "i" or dkind == "f" or (dkind == "u" and data.dtype.itemsize == 1)
        ):
            raise ValueError("Unsupported data type '%s'" % data.dtype)

        header_data = b""

        header_data += b"RIFF"
        header_data += b"\x00\x00\x00\x00"
        header_data += b"WAVE"

        # fmt chunk
        header_data += b"fmt "
        if dkind == "f":
            format_tag = WaveFormat.IEEE_FLOAT
        else:
            format_tag = WaveFormat.PCM
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]
        bit_depth = data.dtype.itemsize * 8
        bytes_per_second = fs * (bit_depth // 8) * channels
        block_align = channels * (bit_depth // 8)

        fmt_chunk_data = struct.pack(
            "<HHIIHH",
            format_tag,
            channels,
            fs,
            bytes_per_second,
            block_align,
            bit_depth,
        )
        if not dkind in ('i', 'u'):
            # add cbSize field for non-PCM files
            fmt_chunk_data += b"\x00\x00"

        header_data += struct.pack("<I", len(fmt_chunk_data))
        header_data += fmt_chunk_data

        # fact chunk (non-PCM files)
        if not dkind in ('i', 'u'):
            header_data += b"fact"
            header_data += struct.pack("<II", 4, data.shape[0])

        # check data size (needs to be immediately before the data chunk)
        if ((len(header_data) - 4 - 4) + (4 + 4 + data.nbytes)) > 0xFFFFFFFF:
            raise ValueError("Data exceeds wave file size limit")

        fid.write(header_data)

        # data chunk
        fid.write(b"data")
        fid.write(struct.pack("<I", data.nbytes))
        if data.dtype.byteorder == ">" or (
            data.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            data = data.byteswap()
        # ravel gives a c-contiguous buffer
        fid.write(data.ravel().view("b").data)

        # Determine file size and place it in correct
        #  position at start of the file.
        size = fid.tell()
        fid.seek(4)
        fid.write(struct.pack("<I", size - 8))

    finally:
        if not hasattr(file, "write"):
            fid.close()
        else:
            fid.seek(0)


PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])


def pin_memory(data):
    string_classes = (str, bytes)
    if isinstance(data, np.ndarray):
        return data.pin_memory()
    if isinstance(data, string_classes):
        return data
    if isinstance(data, collections.abc.Mapping):
        return {k: pin_memory(sample) for k, sample in data.items()}
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(pin_memory(sample) for sample in data))
    if isinstance(data, collections.abc.Sequence):
        return [pin_memory(sample) for sample in data]
    if hasattr(data, "pin_memory"):
        return data.pin_memory()
    return data


def recursive_to(data, *args, **kwargs):
    """Moves data to device, or other type, and handles containers.

    Very similar to pin_memory, but applies .to() instead.
    """
    if isinstance(data, np.ndarray):
        return data.to(*args, **kwargs)
    if isinstance(data, collections.abc.Mapping):
        return {k: recursive_to(sample, *args, **kwargs) for k, sample in data.items()}
    if isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return type(data)(*(recursive_to(sample, *args, **kwargs) for sample in data))
    if isinstance(data, collections.abc.Sequence):
        return [recursive_to(sample, *args, **kwargs) for sample in data]
    if hasattr(data, "to"):
        return data.to(*args, **kwargs)
    # What should be done with unknown data?
    # For now, just return as they are
    return data
