"""Check startup guards without importing or running hardware qualification code."""
import ast
from pathlib import Path
import unittest


class QualificationOptimizationTests(unittest.TestCase):
    def test_assert_based_scripts_require_normal_python_before_imports(self):
        root=Path(__file__).resolve().parents[1]/'scripts'
        checked=0
        for path in sorted(root.glob('*.py')):
            if not path.name.startswith(('verify_','benchmark_','compare_')):continue
            tree=ast.parse(path.read_text(),filename=str(path))
            if not any(isinstance(node,ast.Assert) for node in ast.walk(tree)):continue
            checked+=1
            with self.subTest(script=path.name):
                offset=0
                if (isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant)
                        and isinstance(tree.body[0].value.value,str)):offset=1
                while (isinstance(tree.body[offset],ast.ImportFrom)
                       and tree.body[offset].module=='__future__'):offset+=1
                guard=tree.body[offset]
                self.assertIsInstance(guard,ast.If,'assert-based qualification needs a pre-import optimization guard')
                self.assertEqual(ast.dump(guard.test),ast.dump(ast.parse('not __debug__',mode='eval').body))
                self.assertFalse(guard.orelse)
                self.assertEqual(len(guard.body),1)
                failure=guard.body[0]
                self.assertIsInstance(failure,ast.Raise)
                self.assertIsNone(failure.cause)
                self.assertIsInstance(failure.exc,ast.Call)
                self.assertEqual(ast.dump(failure.exc.func),ast.dump(ast.Name(id='RuntimeError',ctx=ast.Load())))
                self.assertFalse(failure.exc.keywords)
                self.assertTrue(all(isinstance(arg,ast.Constant) and isinstance(arg.value,str) for arg in failure.exc.args))
                # Execute only the guard, never the imports/main body or its GPU checks.
                prefix=ast.Module(body=[guard],type_ignores=[])
                for optimize in (0,1,2):
                    code=compile(prefix,str(path),'exec',optimize=optimize)
                    if optimize:
                        with self.assertRaisesRegex(RuntimeError,'without -O'):exec(code,{})
                    else:exec(code,{})
        self.assertGreater(checked,0)
