'''
variant.py
This command annotates variants in a VCF file using ECASP. It reads the input VCF file, annotates
each variant with delta scores and delta positions, and writes the annotated variants to an output VCF file. 
It uses the Annotator class to annotate variants based on the reference genome and annotation provided. The 
annotated variants are written to the output VCF file with the 'ECASP' INFO field containing the delta
scores and delta positions for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). 
''' 

import logging
import os

import numpy as np
import pysam
import torch
from tqdm import tqdm

from openspliceai.rbp.expression import load_rbp_expression
from openspliceai.vcf_utils import LEGACY_SCORE_INFO_KEY, PRIMARY_SCORE_INFO_KEY
from openspliceai.variant.utils import *

# NOTE: if running with gpu, note that cudnn version should be 8.9.6 or higher, numpy <2.0.0

def variant(args):
    print("Running ECASP with 'variant' mode")
    start_time = time.time()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Error handling for required arguments
    if None in [args.input_vcf, args.output_vcf, args.ref_genome, args.annotation, args.model, args.flanking_size]:
        logging.error('Usage: ecasp [-h] [-m [model]] [-f [flanking_size]] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        exit(1)

    # Define arguments
    ref_genome = args.ref_genome
    annotation = args.annotation
    input_vcf = args.input_vcf
    output_vcf = args.output_vcf
    distance = args.distance
    mask = args.mask
    model = args.model
    flanking_size = args.flanking_size
    model_type = args.model_type
    precision = args.precision
    
    print(f'''Running with genome: {ref_genome}, annotation: {annotation}, 
          model(s): {model}, model_type: {model_type}, 
          input: {input_vcf}, output: {output_vcf}, 
          distance: {distance}, mask: {mask}, flanking_size: {flanking_size}, precision: {precision}''')

    # Load optional RBP expression vector
    rbp_context = None
    if args.rbp_expression:
        try:
            rbp_expr = load_rbp_expression(args.rbp_expression)
            rbp_tensor = torch.tensor(rbp_expr.values, dtype=torch.float32).unsqueeze(0)
            rbp_context = {"tensor": rbp_tensor, "names": rbp_expr.names}
            logging.info(f"Loaded RBP vector dim={rbp_expr.dim} from {args.rbp_expression}")
        except (OSError, ValueError) as exc:
            logging.error(f"Failed to read RBP expression vector: {exc}")
            exit(1)

    # Reading input VCF file
    print('\t[INFO] Reading input VCF file')
    try:
        vcf = pysam.VariantFile(input_vcf)
    except (IOError, ValueError) as e:
        logging.error('Error reading input file: {}'.format(e))
        exit(1)

    # Adding annotation to the header
    header = vcf.header
    if LEGACY_SCORE_INFO_KEY in header.info:
        header.info.remove_header(LEGACY_SCORE_INFO_KEY)
    if PRIMARY_SCORE_INFO_KEY not in header.info:
        header.add_line(
            f'##INFO=<ID={PRIMARY_SCORE_INFO_KEY},Number=.,Type=String,Description="ECASP variant '
            'annotation. These include delta scores (DS) and delta positions (DP) for '
            'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
            'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">'
        )

    # Generating output VCF file
    print('\t[INFO] Generating output VCF file')
    os.makedirs(os.path.dirname(output_vcf), exist_ok=True)
    try:
        output = pysam.VariantFile(output_vcf, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('Error generating output VCF file: {}'.format(e))
        exit(1)

    # Setup the Annotator based on reference genome and annotation
    logging.info('Initializing Annotator class')
    try:
        ann = Annotator(ref_genome, annotation, model, model_type, flanking_size, rbp_context=rbp_context, film_strength=getattr(args, "film_strength", 1.0))
    except ValueError as exc:
        logging.error(f"Annotator initialisation failed: {exc}")
        exit(1)

    # Obtain delta score for each variant in VCF
    for record in tqdm(vcf):
        scores = get_delta_scores(record, ann, distance, mask, flanking_size, precision)
        for info_key in (PRIMARY_SCORE_INFO_KEY, LEGACY_SCORE_INFO_KEY):
            if info_key in record.info:
                del record.info[info_key]
        if scores:
            record.info[PRIMARY_SCORE_INFO_KEY] = scores
        output.write(record)

    # Close input and output VCF files
    vcf.close()
    output.close()
    logging.info('Annotation completed and written to output VCF file')
    
    print("--- %s seconds ---" % (time.time() - start_time))
