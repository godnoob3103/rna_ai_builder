#!/bin/bash

DATA="/mnt/a/data/lung_cancer"
REF=~/reference/star_index
GTF=~/reference/Homo_sapiens.GRCh38.113.gtf
ALIGNED=~/aligned/lung_cancer
TRIMMED="$DATA/trimmed"
COUNTS="$DATA/counts"

mkdir -p "$TRIMMED/tumor" "$TRIMMED/normal"
mkdir -p "$ALIGNED/tumor" "$ALIGNED/normal"
mkdir -p "$COUNTS"

process_sample() {
    local sample=$1      # uppercase SRR ID, e.g. SRR24166342
    local label=$2       # tumor or normal (used for output dirs)
    local sample_lower=$(echo "$sample" | tr '[:upper:]' '[:lower:]')
    # map output label to actual input folder name
    local src_folder=$label
    if [ "$label" = "tumor" ]; then src_folder="cancer"; fi

    echo "===== Processing $sample ($label) ====="

    # Trim
    fastp \
        -i "$DATA/$src_folder/${sample_lower}_1.fastq.gz" \
        -I "$DATA/$src_folder/${sample_lower}_2.fastq.gz" \
        -o "$TRIMMED/$label/${sample}_1.fastq.gz" \
        -O "$TRIMMED/$label/${sample}_2.fastq.gz" \
        -h "$TRIMMED/$label/${sample}_report.html" \
        -j "$TRIMMED/$label/${sample}_report.json" \
        --thread 16 2>&1

    # Align
    rm -rf /tmp/STAR_${sample}
    STAR --runThreadN 16 \
        --genomeDir $REF \
        --readFilesIn "$TRIMMED/$label/${sample}_1.fastq.gz" \
                      "$TRIMMED/$label/${sample}_2.fastq.gz" \
        --readFilesCommand zcat \
        --outSAMtype BAM SortedByCoordinate \
        --outFileNamePrefix "$ALIGNED/$label/${sample}_" \
        --outSAMattributes NH HI AS NM \
        --outTmpDir /tmp/STAR_${sample} 2>&1

    # Count exons
    featureCounts \
        -T 16 -p -P -B -f -O \
        -a $GTF \
        -o "$COUNTS/${sample}_exon_counts.txt" \
        "$ALIGNED/$label/${sample}_Aligned.sortedByCoord.out.bam" 2>&1

    echo "===== Done $sample ====="
}

# Tumor samples (50 samples)
for sample in \
    SRR24166342 SRR24166340 SRR24166338 SRR24166336 SRR24166334 \
    SRR24166332 SRR24166330 SRR24166328 SRR24166278 SRR24166276 \
    SRR24166274 SRR24166272 SRR24166270 SRR24166268 SRR24166266 \
    SRR24166264 SRR24166208 SRR24166206 SRR24166204 SRR24166202 \
    SRR24166200 SRR24166198 SRR24166194 SRR24166262 SRR24166260 \
    SRR24166258 SRR24166256 SRR24166254 SRR24166252 SRR24166250 \
    SRR24166248 SRR24166144 SRR24166142 SRR24166140 SRR24166138 \
    SRR24166136 SRR24166134 SRR24166132 SRR24166130 SRR24166326 \
    SRR24166324 SRR24166322 SRR24166320 SRR24166318 SRR24166316 \
    SRR24166314 SRR24166312 SRR24166192 SRR24166190 SRR24166188; do
    process_sample $sample tumor
done

# Normal samples (50 samples)
for sample in \
    SRR24166343 SRR24166341 SRR24166339 SRR24166337 SRR24166335 \
    SRR24166333 SRR24166331 SRR24166329 SRR24166279 SRR24166277 \
    SRR24166275 SRR24166273 SRR24166271 SRR24166269 SRR24166267 \
    SRR24166265 SRR24166209 SRR24166207 SRR24166205 SRR24166203 \
    SRR24166201 SRR24166199 SRR24166195 SRR24166263 SRR24166261 \
    SRR24166259 SRR24166257 SRR24166255 SRR24166253 SRR24166251 \
    SRR24166249 SRR24166145 SRR24166143 SRR24166141 SRR24166139 \
    SRR24166137 SRR24166135 SRR24166133 SRR24166131 SRR24166327 \
    SRR24166325 SRR24166323 SRR24166321 SRR24166319 SRR24166317 \
    SRR24166315 SRR24166313 SRR24166193 SRR24166191 SRR24166189; do
    process_sample $sample normal
done

echo "===== ALL DONE ====="
