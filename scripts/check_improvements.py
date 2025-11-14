#!/usr/bin/env python3
"""
Verification script for project improvements.
Checks that all infrastructure files are in place and valid.
"""
import sys
from pathlib import Path


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and report."""
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists


def check_file_not_empty(path: Path, description: str) -> bool:
    """Check if a file exists and is not empty."""
    if not path.exists():
        print(f"❌ {description}: {path} (not found)")
        return False

    size = path.stat().st_size
    is_valid = size > 0
    status = "✅" if is_valid else "❌"
    print(f"{status} {description}: {path} ({size} bytes)")
    return is_valid


def main():
    """Run all verification checks."""
    project_root = Path(__file__).parent.parent
    all_checks_passed = True

    print("=" * 60)
    print("PROJECT IMPROVEMENTS VERIFICATION")
    print("=" * 60)

    print("\n📁 INFRASTRUCTURE FILES")
    print("-" * 60)

    checks = [
        (project_root / ".gitignore", "Git ignore file"),
        (project_root / "LICENSE", "License file"),
        (project_root / "pyproject.toml", "Project configuration"),
        (project_root / "IMPROVEMENTS.md", "Improvements documentation"),
    ]

    for path, desc in checks:
        if not check_file_not_empty(path, desc):
            all_checks_passed = False

    print("\n🧪 TEST FILES")
    print("-" * 60)

    test_files = [
        (project_root / "tests" / "__init__.py", "Test package init"),
        (project_root / "tests" / "test_config.py", "Config tests"),
        (project_root / "tests" / "test_memory_reader.py", "Memory reader tests"),
        (project_root / "tests" / "test_anti_loop.py", "Anti-loop tests"),
    ]

    for path, desc in test_files:
        if not check_file_not_empty(path, desc):
            all_checks_passed = False

    print("\n⚙️ CI/CD FILES")
    print("-" * 60)

    ci_files = [
        (project_root / ".github" / "workflows" / "ci.yml", "GitHub Actions CI"),
    ]

    for path, desc in ci_files:
        if not check_file_not_empty(path, desc):
            all_checks_passed = False

    print("\n📝 SOURCE FILES (MODIFIED)")
    print("-" * 60)

    source_files = [
        (project_root / "src" / "config.py", "Config module"),
        (project_root / "src" / "hyperparameters.py", "Hyperparameters"),
        (project_root / "README.md", "README documentation"),
    ]

    for path, desc in source_files:
        if not check_file_not_empty(path, desc):
            all_checks_passed = False

    print("\n🔍 CACHE FILES (SHOULD NOT EXIST)")
    print("-" * 60)

    cache_patterns = [
        ("__pycache__", "Python cache directories"),
        ("*.pyc", "Python compiled files"),
        ("*.pyo", "Python optimized files"),
    ]

    for pattern, desc in cache_patterns:
        found = list(project_root.rglob(pattern))
        if found:
            print(f"⚠️  {desc} found: {len(found)} items")
            for item in found[:3]:  # Show first 3
                print(f"   - {item.relative_to(project_root)}")
            if len(found) > 3:
                print(f"   ... and {len(found) - 3} more")
            all_checks_passed = False
        else:
            print(f"✅ {desc}: None found")

    print("\n📊 CONFIG VALIDATION")
    print("-" * 60)

    try:
        sys.path.insert(0, str(project_root))
        from src.config import Config

        config = Config()
        print(f"✅ Config loads successfully")
        print(f"   - Device: {config.device}")
        print(f"   - Frame stack size: {config.frame_stack_size}")
        print(f"   - Save frequency: {config.save_frequency}")
        print(f"   - Anti-loop enabled: {config.anti_loop_enabled}")

        # Check for old uppercase attributes
        if hasattr(config, 'DEVICE'):
            print("⚠️  Old uppercase attributes still present (DEVICE)")
        if hasattr(config, 'FRAME_STACK_SIZE'):
            print("⚠️  Old uppercase attributes still present (FRAME_STACK_SIZE)")

    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        all_checks_passed = False

    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update src/main.py to use snake_case config attributes")
        print("2. Run: pytest tests/ -v")
        print("3. Run: black src tests && isort src tests")
        print("4. Run: flake8 src tests")
        print("5. Commit changes: git add . && git commit -m 'Apply improvements'")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease review the issues above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
