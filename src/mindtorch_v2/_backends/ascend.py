try:
    import acl
    HAS_ACL = True
except Exception:
    HAS_ACL = False


def is_available():
    return HAS_ACL
