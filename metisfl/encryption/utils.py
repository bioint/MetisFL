
def _generate_crypto_params_ckks(self, bsize, sbits) -> None:
    # cosntructt ckks
    ckks_scheme = fhe.CKKS(he_scheme_pb.ckks_scheme_config.batch_size,
                        he_scheme_pb.ckks_scheme_config.scaling_factor_bits)

    # ckks.gneerate()
    # ckks.get_files()